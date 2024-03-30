from typing import List, Optional, Mapping, Any, Literal
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import glob
import os

from dflow_galaxy.core.pydantic import BaseModel
from dflow_galaxy.core.dispatcher import BaseApp, PythonApp, create_dispatcher, ExecutorConfig
from dflow_galaxy.core.dflow import DFlowBuilder
from dflow_galaxy.core.util import bash_iter_ls_slice, safe_ln, bash_ln_cmd, bash_inspect_dir, inspect_dir
from dflow_galaxy.core import types

from ai2_kit.domain.cp2k import make_cp2k_task_dirs
from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.util import cmd_with_checkpoint as cmd_cp, list_sample, load_text, dump_text, ensure_dir

from dflow import argo_range

from .lib import resolve_artifact

SYSTEM_DIR = './system_dir'


class Cp2kApp(BaseApp):
    cp2k_cmd: str = 'cp2k.popt'
    concurrency: int = 5


class Cp2kConfig(BaseModel):
    init_systems: List[str] = []
    input_template: Optional[str] = None
    template_vars: Mapping[str, Any] = {}

    limit: int = 50
    limit_method: Literal['even', 'random', 'truncate'] = 'even'


@dataclass(frozen=True)
class SetupCp2kTasksArgs:
    system_dir: types.InputArtifact
    work_dir: types.OutputArtifact


class SetupCp2kTaskFn:

    def __init__(self, config: Cp2kConfig, systems: Mapping[str, Artifact], init: bool):
        self.config = config
        self.init = init
        self.systems = systems

    def __call__(self, args: SetupCp2kTasksArgs):
        safe_ln(args.system_dir, SYSTEM_DIR)
        inspect_dir(SYSTEM_DIR)

        limit = self.config.limit

        system_files: List[ArtifactDict] = []
        if self.init:
            # handle init systems
            limit = 0  # no limit for init systems
            assert self.config.init_systems, 'init_systems should not be empty for first iteration'
            for k in self.config.init_systems:
                v = deepcopy(self.systems[k])  # avoid side effect
                v.url = os.path.join(SYSTEM_DIR, k)
                system_files.extend(resolve_artifact(v))
        else:
            # handle iter systems
            system_dirs = glob.glob(f'{SYSTEM_DIR}/system/*')  # search pattern is defined by model_devi
            for sys_dir in system_dirs:
                sys_dir = Path(sys_dir)
                ancestor = load_text(sys_dir / 'ANCESTOR')
                a_dict = {
                    'url': str(sys_dir / 'decent.xyz'),
                    'format': 'extxyz',
                    'attrs': deepcopy(self.systems[ancestor].attrs),
                }
                system_files.append(a_dict)  # type: ignore

        assert system_files, 'no system files found'

        task_dir = os.path.join(args.work_dir, 'tasks')
        ensure_dir(task_dir)
        task_dirs = make_cp2k_task_dirs(
            system_files=system_files,
            input_template=self.config.input_template,
            template_vars=self.config.template_vars,
            base_dir=task_dir,
            limit=limit,
            limit_method=self.config.limit_method,
            # not supported yet
            mode='default',
            wfn_warmup_template=None,
            # TODO: type_map is no longer needed, should be fixed in ai2-kit
            type_map=[],
        )
        for task_dir in task_dirs:
            path = os.path.join(task_dir['url'], 'ANCESTOR')
            dump_text(task_dir['attrs']['ancestor'], path)


@dataclass(frozen=True)
class RunCp2kTasksArgs:
    slice_idx: types.InputParam[types.SliceIndex]
    work_dir: types.InputArtifact
    persist_dir: types.OutputArtifact


class RunCp2kTasksFn:
    def __init__(self, config: Cp2kConfig, context: Cp2kApp):
        self.config = config
        self.context = context

    def __call__(self, args: RunCp2kTasksArgs):
        c = self.context.concurrency

        script = [
            f'mkdir -p {args.persist_dir}',
            f"pushd {args.work_dir}",
            bash_inspect_dir('.'),
            bash_iter_ls_slice(
                'tasks/*/', opt='-d', n=c, i=args.slice_idx, it_var='ITEM',
                script=[
                    'pushd $ITEM',
                    'mv persist/* . || true  # recover checkpoint',
                    '',
                    self._build_cp2k_script(),
                    '',
                    '# persist result',
                    f'PERSIST_DIR={args.persist_dir}/$ITEM/persist/',
                    'mkdir -p $PERSIST_DIR',
                    'mv *.done output ANCESTOR $PERSIST_DIR',
                    'popd',
                ]
            ),
            'popd',
        ]
        return script

    def _build_cp2k_script(self):
        cmd = f'''{self.context.cp2k_cmd} -i input.inp &> output'''
        return cmd_cp(cmd, 'cp2k.done')


def provision_cp2k(builder: DFlowBuilder, ns: str, /,
                   config: Cp2kConfig,
                   executor: ExecutorConfig,
                   cp2k_app: Cp2kApp,
                   python_app: PythonApp,

                   system_url: str,
                   work_dir_url: str,

                   init: bool,
                   systems: Mapping[str, Artifact],
                   ):
    setup_tasks_fn = SetupCp2kTaskFn(config, systems=systems, init=init)
    setup_tasks_step = builder.make_python_step(setup_tasks_fn, uid=f'{ns}-setup-task',
                                                setup_script=python_app.setup_script,
                                                executor=create_dispatcher(executor, python_app.resource))(
        SetupCp2kTasksArgs(
            system_dir=system_url,
            work_dir=work_dir_url,
        )
    )

    run_tasks_fn = RunCp2kTasksFn(config, cp2k_app)
    run_tasks_step = builder.make_bash_step(run_tasks_fn, uid=f'{ns}-run-task',
                                            setup_script=cp2k_app.setup_script,
                                            with_param=argo_range(cp2k_app.concurrency),
                                            executor=create_dispatcher(executor, cp2k_app.resource))(
        RunCp2kTasksArgs(
            slice_idx='{{item}}',
            work_dir=work_dir_url,
            persist_dir=work_dir_url,
        )
    )

    builder.add_step(setup_tasks_step)
    builder.add_step(run_tasks_step)
