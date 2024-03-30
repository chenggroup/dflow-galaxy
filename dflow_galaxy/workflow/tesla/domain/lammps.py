from typing import List, Optional, Mapping, Any, Literal
from dataclasses import dataclass
from copy import deepcopy
import glob
import os

from dflow_galaxy.core.pydantic import BaseModel
from dflow_galaxy.core.dispatcher import BaseApp, PythonApp, create_dispatcher, ExecutorConfig
from dflow_galaxy.core.dflow import DFlowBuilder
from dflow_galaxy.core.util import bash_iter_ls_slice, safe_ln, bash_ln_cmd, inspect_dir, bash_inspect_dir
from dflow_galaxy.core import types

from ai2_kit.domain.lammps import make_lammps_task_dirs, FepOptions
from ai2_kit.domain.constant import DP_FROZEN_MODEL
from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.util import cmd_with_checkpoint as cmd_cp, dump_text

from dflow import argo_range


from .lib import resolve_artifact


MODEL_DIR = './mlp-models'
SYSTEM_DIR = './systems'


class LammpsApp(BaseApp):
    lammps_cmd: str = 'lmp'
    concurrency: int = 5


class LammpsConfig(BaseModel):
    systems: List[str]

    nsteps: int
    no_pbc: bool = False
    timestep: float = 0.0005
    sample_freq: int = 100

    product_vars: Mapping[str, List[Any]]
    broadcast_vars: Mapping[str, Any] = {}

    template_vars: Mapping[str, Any] = {}
    """
    input_template may provide extra injection points for user to inject custom settings.
    Those value could be set here.

    Those vars can be referenced in the LAMMPS input template as $$VAR_NAME.
    """

    input_template: Optional[str] = None
    plumed_config: Optional[str] = None
    ensemble: Optional[Literal['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve', 'csvr']] = None
    ignore_error: bool = False


@dataclass(frozen=True)
class SetupLammpsTasksArgs:
    model_dir: types.InputArtifact
    system_dir: types.InputArtifact
    work_dir: types.OutputArtifact


class SetupLammpsTasksFn:
    def __init__(self, config: LammpsConfig,
                 type_map: List[str],
                 mass_map: List[float],
                 systems: Mapping[str, Artifact]):
        self.config = config
        self.type_map = type_map
        self.mass_map = mass_map
        self.systems = systems

    def __call__(self, args: SetupLammpsTasksArgs):
        # dflow didn't provide a unified file namespace,
        # so we have to link dataset to a fixed path and use relative path to access it
        safe_ln(args.model_dir, MODEL_DIR)
        inspect_dir(MODEL_DIR)
        safe_ln(args.system_dir, SYSTEM_DIR)
        inspect_dir(SYSTEM_DIR)

        # resolve input data
        data_files: List[ArtifactDict] = []
        for k in self.config.systems:
            v = deepcopy(self.systems[k])  # avoid side effect
            v.url = os.path.join(SYSTEM_DIR, k)
            data_files.extend(resolve_artifact(v))

        # resolve model files
        search_pattern = f'{MODEL_DIR}/tasks/*/persist/{DP_FROZEN_MODEL}'
        model_files = glob.glob(search_pattern)
        assert model_files, f'no model files found in {search_pattern}'

        # handle default value
        default_vars = {
            'TAU_T': [0.1],
            'TAU_P': [0.5],
            'TIME_CONST': [0.1],
        }
        for k, v in default_vars.items():
            if k not in self.config.product_vars and k not in self.config.broadcast_vars:
                self.config.broadcast_vars[k] = v  # type: ignore

        _base_dir, task_dirs = make_lammps_task_dirs(
            combination_vars=self.config.product_vars,
            broadcast_vars=self.config.broadcast_vars,
            data_files=data_files,
            dp_models={'': model_files},
            n_steps=self.config.nsteps,
            timestep=self.config.timestep,
            sample_freq=self.config.sample_freq,
            no_pbc=self.config.no_pbc,
            ensemble=self.config.ensemble,
            input_template=self.config.input_template,
            plumed_config=self.config.plumed_config,
            extra_template_vars=self.config.template_vars,
            type_map=self.type_map,
            mass_map=self.mass_map,
            work_dir=args.work_dir,

            # TODO: support more feature in the future
            mode='default',
            type_alias={},
            dp_modifier=None,
            dp_sel_type=None,
            preset_template='default',
            n_wise=0,
            fix_statement=None,
            rel_path=True,
            fep_opts=FepOptions(),
        )
        # write ancestor to the task dirs
        for task_dir in task_dirs:
            path = os.path.join(task_dir['url'], 'ANCESTOR')
            dump_text(task_dir['attrs']['ancestor'], path)


@dataclass(frozen=True)
class RunLammpsTasksArgs:
    slice_idx: types.InputParam[types.SliceIndex]
    model_dir: types.InputArtifact
    work_dir: types.InputArtifact
    persist_dir: types.OutputArtifact


class RunLammpsTasksFn:
    def __init__(self, config: LammpsConfig, context: LammpsApp):
        self.config = config
        self.context = context

    def __call__(self, args: RunLammpsTasksArgs):
        c = self.context.concurrency

        script = [
            f'mkdir -p {args.persist_dir}',
            bash_inspect_dir(args.work_dir),
            f"pushd {args.work_dir}",
            bash_iter_ls_slice(
                'tasks/*/', opt='-d', n=c, i=args.slice_idx, it_var='ITEM',
                script=[
                    '# run lammps',
                    'pushd $ITEM',
                    bash_ln_cmd(args.model_dir, MODEL_DIR),
                    'mv persist/* . || true  # restore previous state',
                    '',
                    self._build_lammps_cmd(),
                    '',
                    '# persist result',
                    f'PERSIST_DIR={args.persist_dir}/$ITEM/persist/',
                    'mkdir -p $PERSIST_DIR',
                    'mv *.done traj model_devi.out ANCESTOR $PERSIST_DIR',
                    'popd',
                ]
            ),
            'popd',
        ]
        return script

    def _build_lammps_cmd(self):
        lmp_cmd = f'{self.context.lammps_cmd} -i lammps.input'
        cmd = f'''if [ -f md.restart.* ]; then {lmp_cmd} -v restart 1; else {lmp_cmd} -v restart 0; fi'''
        return cmd_cp(cmd, 'lammps.done', ignore_error=self.config.ignore_error)


def provision_lammps(builder: DFlowBuilder, ns: str, /,
                     config: LammpsConfig,
                     executor: ExecutorConfig,
                     lammps_app: LammpsApp,
                     python_app: PythonApp,

                     systems_url: str,
                     mlp_model_url: str,
                     work_dir_url: str,
                     type_map: List[str],
                     mass_map: List[float],
                     systems: Mapping[str, Artifact],
                     ):
    setup_tasks_fn = SetupLammpsTasksFn(config, type_map=type_map, mass_map=mass_map, systems=systems)
    setup_tasks_step = builder.make_python_step(setup_tasks_fn, uid=f'{ns}-setup-task',
                                                setup_script=python_app.setup_script,
                                                executor=create_dispatcher(executor, python_app.resource))(
        SetupLammpsTasksArgs(
            model_dir=mlp_model_url,
            system_dir=systems_url,
            work_dir=work_dir_url,
        )
    )

    run_tasks_fn = RunLammpsTasksFn(config, lammps_app)
    run_tasks_step = builder.make_bash_step(run_tasks_fn, uid=f'{ns}-run-task',
                                            setup_script=lammps_app.setup_script,
                                            with_param=argo_range(lammps_app.concurrency),
                                            executor=create_dispatcher(executor, lammps_app.resource))(
        RunLammpsTasksArgs(
            slice_idx='{{item}}',
            model_dir=mlp_model_url,
            work_dir=work_dir_url,
            persist_dir=work_dir_url,
        )
    )

    builder.add_step(setup_tasks_step)
    builder.add_step(run_tasks_step)
