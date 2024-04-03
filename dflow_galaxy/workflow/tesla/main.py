from typing import Optional
from copy import deepcopy

from ai2_kit.core.util import load_yaml_files, merge_dict
from ai2_kit.core.cmd import CmdGroup

from dflow_galaxy.core.dflow import DFlowBuilder
from dflow_galaxy.core.util import not_none
from dflow_galaxy.core.log import get_logger

from .config import TeslaConfig, WorkflowConfig
from .domain import deepmd, lammps, model_devi, cp2k
from .domain.lib import StepSwitch, LabelApp, ExploreApp

logger = get_logger(__name__)

class RuntimeContext:
    train_url: Optional[str]

    explore_url: Optional[str]
    explore_app: ExploreApp

    screen_url: Optional[str]

    label_url: Optional[str]
    label_app: LabelApp


def run_tesla(*config_files: str, s3_prefix: str, debug: bool = False, skip: bool = False, max_iters: int = 1):
    config_raw = load_yaml_files(*config_files)
    config = TeslaConfig(**config_raw)
    config.init()

    builder = DFlowBuilder(name='tesla', s3_prefix=s3_prefix, debug=debug,
                           default_archive=None)
    step_switch = StepSwitch(skip)
    runtime_ctx = RuntimeContext()

    raw_workflow_cfg = config.workflow
    for iter_num in range(max_iters):
        workflow_cfg = WorkflowConfig(**raw_workflow_cfg)

        type_map = workflow_cfg.general.type_map
        mass_map = workflow_cfg.general.mass_map

        iter_str = f'{iter_num:03d}'

        # Labeling
        cp2k_cfg = workflow_cfg.label.cp2k
        if cp2k_cfg:
            step_name = f'label-cp2k-iter-{iter_str}'
            runtime_ctx.label_app = 'cp2k'
            runtime_ctx.label_url = f's3://./label-cp2k/iter/{iter_str}'

            cp2k_executor = not_none(config.executors[not_none(config.orchestration.cp2k)])
            if (iter_num > 0 or cp2k_cfg.init_systems) and \
                    not step_switch.shall_skip(step_name):
                if iter_num == 0:
                    assert cp2k_cfg.init_systems, 'init_systems should not be empty for first iteration'
                    assert runtime_ctx.screen_url is None, f'explore_url should be None for iter 0, actual: {runtime_ctx.screen_url}'

                for sys_key in cp2k_cfg.init_systems:
                    sys = not_none(config.datasets[sys_key])
                    builder.s3_upload(sys.url, f'init-systems/{sys_key}', cache=True)
                cp2k.provision_cp2k(builder, step_name,
                                    config=cp2k_cfg,
                                    executor=cp2k_executor,
                                    cp2k_app=not_none(cp2k_executor.apps.cp2k),
                                    python_app=not_none(cp2k_executor.apps.python),

                                    system_url=runtime_ctx.screen_url or 's3://./init-systems',
                                    work_dir_url=runtime_ctx.label_url,

                                    init=(iter_num == 0),
                                    systems=config.datasets,)

        # Training
        deepmd_cfg = workflow_cfg.train.deepmd
        if deepmd_cfg:
            step_name = f'train-deepmd-iter-{iter_str}'
            runtime_ctx.train_url = f's3://./train-deepmd/iter/{iter_str}'
            deepmd_executor = not_none(config.executors[not_none(config.orchestration.deepmd)])

            if not step_switch.shall_skip(step_name):
                for ds_key in deepmd_cfg.init_dataset:
                    ds = not_none(config.datasets[ds_key])
                    builder.s3_upload(ds.url, f'init-dataset/{ds_key}', cache=True)  # set cache to avoid re-upload
                deepmd.provision_deepmd(builder, step_name,
                                        config=deepmd_cfg,
                                        executor=deepmd_executor,
                                        deepmd_app=not_none(deepmd_executor.apps.deepmd),
                                        python_app=not_none(deepmd_executor.apps.python),

                                        label_app=runtime_ctx.label_app,
                                        label_dir_url=runtime_ctx.label_url,

                                        init_dataset_url='s3://./init-dataset',
                                        iter_dataset_url='s3://./iter-dataset',
                                        work_dir_url=runtime_ctx.train_url,
                                        iter_str=iter_str,
                                        type_map=type_map)

        else:
            raise ValueError('No training app specified')

        # Exploration
        lammps_cfg = workflow_cfg.explore.lammps
        if lammps_cfg:
            step_name = f'explore-lammps-iter-{iter_str}'
            runtime_ctx.explore_url = f's3://./explore-lammps/iter/{iter_str}'
            runtime_ctx.explore_app = 'lammps'

            lammps_executor = not_none(config.executors[not_none(config.orchestration.lammps)])

            if not step_switch.shall_skip(step_name):
                for sys_key in lammps_cfg.systems:
                    sys = not_none(config.datasets[sys_key])
                    builder.s3_upload(sys.url, f'explore-systems/{sys_key}', cache=True)

                lammps.provision_lammps(builder, step_name,
                                        config=lammps_cfg,
                                        executor=lammps_executor,
                                        lammps_app=not_none(lammps_executor.apps.lammps),
                                        python_app=not_none(lammps_executor.apps.python),

                                        mlp_model_url=runtime_ctx.train_url,
                                        systems_url='s3://./explore-systems',
                                        work_dir_url=runtime_ctx.explore_url,
                                        type_map=type_map,
                                        mass_map=mass_map,
                                        systems=config.datasets)
        else:
            raise ValueError('No explore app specified')

        # Screening
        model_devi_cfg = workflow_cfg.screen.model_devi
        if model_devi_cfg:
            step_name = f'screen-model-devi-iter-{iter_str}'
            runtime_ctx.screen_url = f's3://./screen-model-devi/iter/{iter_str}'
            model_devi_executor = not_none(config.executors[not_none(config.orchestration.model_devi)])

            if not step_switch.shall_skip(step_name):
                model_devi.provision_model_devi(builder, step_name,
                                                config=model_devi_cfg,
                                                executor=model_devi_executor,
                                                python_app=not_none(model_devi_executor.apps.python),

                                                explore_app=runtime_ctx.explore_app,
                                                explore_data_url=runtime_ctx.explore_url,

                                                persist_data_url=runtime_ctx.screen_url,
                                                type_map=type_map)
        else:
            raise ValueError('No screen app specified')

        if workflow_cfg.update:
            if iter_num == workflow_cfg.update.until_iter:
                logger.info('Updating workflow config at iter %d', iter_num)
                # the patch is applied to the original config, not the updated one
                raw_workflow_cfg = deepcopy(config.workflow)
                raw_workflow_cfg['workflow']['update'] = None  # clean the old update config
                raw_workflow_cfg = merge_dict(raw_workflow_cfg, workflow_cfg.update.patch)

    builder.run()


cmd_entry = CmdGroup({
    'run': run_tesla,
}, doc='TESLA workflow')
