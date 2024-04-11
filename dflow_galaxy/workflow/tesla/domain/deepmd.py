from dataclasses import dataclass
from typing import List, Tuple, Optional
from itertools import groupby
from pathlib import Path
import glob

from ai2_kit.domain.deepmd import make_deepmd_task_dirs
from ai2_kit.core.util import cmd_with_checkpoint as cmd_cp, load_text
from ai2_kit.domain.constant import DP_INPUT_FILE, DP_ORIGINAL_MODEL, DP_FROZEN_MODEL

from dflow_galaxy.core.pydantic import BaseModel
from dflow_galaxy.core.dispatcher import BaseApp, PythonApp, create_dispatcher, ExecutorConfig
from dflow_galaxy.core.dflow import DFlowBuilder
from dflow_galaxy.core.util import bash_iter_ls_slice, safe_ln, bash_ln_cmd, bash_inspect_dir, inspect_dir
from dflow_galaxy.core.log import get_logger
from dflow_galaxy.core import types

from dflow import argo_range
import dpdata

from .lib import LabelApp

logger = get_logger(__name__)

INIT_DATASET_DIR = './init-dataset'
ITER_DATASET_DIR = './iter-dataset'

class DeepmdApp(BaseApp):
    dp_cmd: str = 'dp'
    concurrency: int = 4


class DeepmdConfig(BaseModel):
    model_num: int = 4
    init_dataset: List[str] = []
    input_template: dict = {}
    compress_model: bool = False
    ignore_error: bool = False


@dataclass(frozen=True)
class UpdateDatasetArgs:
    label_dir: types.InputArtifact
    iter_dataset_dir: types.OutputArtifact


class UpdateDatasetFn:
    def __init__(self, config: DeepmdConfig, iter_str: str, label_app: Optional[LabelApp], type_map: List[str]):
        self.config = config
        self.iter_str = iter_str
        self.label_app = label_app
        self.type_map = type_map

    def __call__(self, args: UpdateDatasetArgs):
        dp_sys_list:List[Tuple[str, dpdata.LabeledSystem]] = []

        # parse label data
        if self.label_app == 'cp2k':
            cp2k_dirs = glob.glob(f'{args.label_dir}/tasks/*/persist')
            for cp2k_dir in cp2k_dirs:
                try:
                    ancestor = load_text(f'{cp2k_dir}/ANCESTOR')
                    dp_sys = dpdata.LabeledSystem(f'{cp2k_dir}/output', fmt='cp2k/output', type_map=self.type_map)
                    if dp_sys is not None and len(dp_sys) > 0:
                        dp_sys_list.append((ancestor,dp_sys))
                    else:
                        logger.warn(f'Ignore empty dp system: {cp2k_dir}')
                except Exception as e:
                    logger.exception(f'Failed to load cp2k output: {cp2k_dir}')
                    if not self.config.ignore_error:
                        raise e
        else:
            raise ValueError(f'Unsupported label app: {self.label_app}')

        # dump data
        dataset_dir = Path(args.iter_dataset_dir) / self.iter_str
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # group dataset by ancestor
        dp_sys_list = sorted(dp_sys_list, key=lambda x: x[0])  # sorted by ancestor
        for ancestor, group in groupby(dp_sys_list, key=lambda x: x[0]):
            group = list(group)
            assert ancestor, 'ancestor should not be empty'
            dataset = group[0][1]
            for _a, dp_sys in group[1:]:
                dataset += dp_sys
            dataset.to_deepmd_npy(dataset_dir / ancestor, set_size=len(dataset), type_map=self.type_map)  # type: ignore


@dataclass(frozen=True)
class SetupDeepmdTasksArgs:
    init_dataset_dir: types.InputArtifact
    iter_dataset_dir: types.InputArtifact

    work_dir: types.OutputArtifact


class SetupDeepmdTaskFn:
    def __init__(self, config: DeepmdConfig, type_map: List[str]):
        self.config = config
        self.type_map = type_map

    def __call__(self, args: SetupDeepmdTasksArgs):
        # dflow didn't provide a unified file namespace,
        # so we have to link dataset to a fixed path and use relative path to access it
        safe_ln(args.init_dataset_dir, INIT_DATASET_DIR)
        inspect_dir(INIT_DATASET_DIR)
        safe_ln(args.iter_dataset_dir, ITER_DATASET_DIR)
        inspect_dir(ITER_DATASET_DIR)

        train_dataset_dirs = [ f'{INIT_DATASET_DIR}/{ds}' for ds in self.config.init_dataset]
        train_dataset_dirs.extend(glob.glob(f'{ITER_DATASET_DIR}/*/*'))

        make_deepmd_task_dirs(input_template=self.config.input_template,
                              model_num=self.config.model_num,
                              train_systems=train_dataset_dirs,
                              type_map=self.type_map,
                              base_dir=f'{args.work_dir}/tasks',
                              # TODO: support the following parameters
                              isolate_outliers=False,
                              validation_systems=[],
                              outlier_systems=[],
                              outlier_weight=-1.0,
                              dw_input_template=None,
                              )


@dataclass(frozen=True)
class RunDeepmdTrainingArgs:
    slice_idx: types.InputParam[types.SliceIndex]

    init_dataset_dir: types.InputArtifact
    iter_dataset_dir: types.InputArtifact

    work_dir: types.InputArtifact
    persist_dir: types.OutputArtifact


class RunDeepmdTrainingFn:
    def __init__(self, config: DeepmdConfig, context: DeepmdApp):
        self.config = config
        self.context = context

    def __call__(self, args: RunDeepmdTrainingArgs):
        """generate bash script to run deepmd training commands"""
        c = self.context.concurrency

        script = [
            f'mkdir -p {args.persist_dir}',
            bash_inspect_dir(args.work_dir),
            f"pushd {args.work_dir}",
            bash_iter_ls_slice(
                'tasks/*/', opt='-d', n=c, i=args.slice_idx, it_var='ITEM',
                script=[
                    '# dp train',
                    'pushd $ITEM',
                    'mv persist/* . || true  # recover checkpoint',
                    bash_ln_cmd(args.init_dataset_dir, INIT_DATASET_DIR),
                    bash_ln_cmd(args.iter_dataset_dir, ITER_DATASET_DIR),
                    '',
                    self._build_dp_train_script(),
                    '',
                    '# persist result',
                    f'PERSIST_DIR={args.persist_dir}/$ITEM/persist/',
                    'mkdir -p $PERSIST_DIR',
                    'rm model.ckpt* || true'
                    'mv *.* $PERSIST_DIR',
                    'popd',
                ]
            ),
            'popd',
        ]
        return script

    def _build_dp_train_script(self):
        dp_cmd = self.context.dp_cmd
        train_cmd = f'{dp_cmd} train {DP_INPUT_FILE}'
        # TODO: handle restart, initialize from previous model, support pretrain model
        script = [
            cmd_cp(train_cmd, 'dp-train.done'),
            cmd_cp(f'{dp_cmd} freeze -o {DP_ORIGINAL_MODEL}', 'dp-freeze.done'),
        ]
        # compress (optional) and frozen model
        if self.config.compress_model:
            freeze_cmd = f'{dp_cmd} compress -i {DP_ORIGINAL_MODEL} -o {DP_FROZEN_MODEL}'
        else:
            freeze_cmd = f'mv {DP_ORIGINAL_MODEL} {DP_FROZEN_MODEL}'
        script.append(cmd_cp(freeze_cmd, 'dp-compress.done'))

        return '\n'.join(script)


def provision_deepmd(builder: DFlowBuilder, ns: str, /,
                     config: DeepmdConfig,
                     executor: ExecutorConfig,
                     deepmd_app: DeepmdApp,
                     python_app: PythonApp,
                     work_dir_url: str,

                     label_app: Optional[LabelApp],
                     label_dir_url: Optional[str],

                     init_dataset_url: str,
                     iter_dataset_url: str,
                     iter_str: str,
                     type_map: List[str],
                     ):
    if label_app and label_dir_url:
        update_dataset_fn = UpdateDatasetFn(config, iter_str=iter_str,
                                            label_app=label_app, type_map=type_map)
        update_dataset_step = builder.make_python_step(update_dataset_fn, uid=f'{ns}-update-dataset',
                                                       setup_script=python_app.setup_script,
                                                       executor=create_dispatcher(executor, python_app.resource))(
            UpdateDatasetArgs(
                label_dir=label_dir_url,
                iter_dataset_dir=iter_dataset_url,
            )

        )
        builder.add_step(update_dataset_step)

    setup_tasks_fn = SetupDeepmdTaskFn(config, type_map)
    setup_tasks_step = builder.make_python_step(setup_tasks_fn, uid=f'{ns}-setup-task',
                                                setup_script=python_app.setup_script,
                                                executor=create_dispatcher(executor, python_app.resource))(
        SetupDeepmdTasksArgs(
            init_dataset_dir=init_dataset_url,
            iter_dataset_dir=iter_dataset_url,
            work_dir=work_dir_url,
        )
    )
    run_training_fn = RunDeepmdTrainingFn(config=config, context=deepmd_app)
    run_training_step = builder.make_bash_step(run_training_fn, uid=f'{ns}-run-training',
                                               setup_script=deepmd_app.setup_script,
                                               with_param=argo_range(deepmd_app.concurrency),
                                               executor=create_dispatcher(executor, deepmd_app.resource))(
        RunDeepmdTrainingArgs(
            slice_idx="{{item}}",
            init_dataset_dir=init_dataset_url,
            iter_dataset_dir=iter_dataset_url,
            work_dir=work_dir_url,
            persist_dir=work_dir_url,
        )
    )

    builder.add_step(setup_tasks_step)
    builder.add_step(run_training_step)
