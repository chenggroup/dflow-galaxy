from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath
from dp.launching.typing import Int, String, Enum, Float, Boolean, List, Optional, Dict
from dp.launching.cli import to_runner, default_minimal_exception_handler

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context, EnsembleOptions
from dflow_galaxy.res import get_res_path
from dflow_galaxy.core.log import get_logger

from dflow_galaxy.workflow.tesla.main import build_tesla_workflow

from ai2_kit.core.util import dump_json, load_text, load_json
from ai2_kit.feat import catalysis as ai2cat

import ase.io

from pathlib import Path
from uuid import uuid4
import shutil
import sys
import os


logger = get_logger(__name__)


class KvItem(BaseModel):
    key: String = Field(
        description="Key of the item")
    value: String = Field(
        format='multi-line',
        description="Value of the item")


class ExploreItem(BaseModel):
    key: String = Field(
        description="Key of the item")
    value: String = Field(
        description="Value of the item, multiple value should be separated by comma")
    broadcast: Boolean = Field(
        default=False,
        description="Use broadcast instead of full combination")


class DeepmdSettings(BaseModel):
    dataset : InputFilePath = Field(
        description="DeepMD dataset folder in npy format")

    input_template: InputFilePath = Field(
        description="Input template file for DeepMD training")

    compress_model: Boolean = Field(
        default=True,
        description="Compress the model after training")

    concurrency: Int = Field(
        default=4,
        description="Number of concurrent run")

    resource: Dict[String, String] = Field(
        default={
            'image_name': 'registry.dp.tech/dptech/dpmd:2.2.8-cuda11.8',
            'scass_type': 'c8_m32_1 * NVIDIA V100',
        },
        description='Bohrium resource for DeepMD training, no need to change in most cases',
    )

    cmd: String = Field(
        default='dp',
        description="Command to run DeepMD, note that it depends on the docker image you used")


class LammpsSetting(BaseModel):
    system_file: InputFilePath = Field(
        description="Structure file in extxyz format use for LAMMPS simulation")

    ensemble: EnsembleOptions = Field(
        default=EnsembleOptions.csvr,
        description='Ensemble of LAMMPS simulation')

    plumed_config: String = Field(
        description='Plumed configuration file for metadynamics simulation')

    explore_vars: List[ExploreItem] = Field(
        default=[
           ExploreItem(key='TEMP', value='50,100,200,300,400,600,800,1000', broadcast=False),
           ExploreItem(key='PRES', value='1', broadcast=False),
        ],
        description="Variables for LAMMPS exploration, TEMP, PRES are required, you may also add TAU_T, TAU_P, etc")

    template_vars: List[KvItem] = Field(
        default=[
            KvItem(key='POST_INIT', value='\n'.join([
                'neighbor 1.0 bin',
                'box      tilt large',
            ])),
            KvItem(key='POST_READ_DATA', value='\n'.join([
                'change_box all triclinic'
            ])),
        ],
        description="Template variables for LAMMPS exploration, you may need to modify this according to your system")

    nsteps: Int = Field(
        default=1000,
        description='Number of steps of LAMMPS simulation for the first iteration')

    timestep: Float = Field(
        default=0.0005,
        description='Time step size of LAMMPS simulation in ps')

    sample_freq: Int = Field(
        default=10,
        description='Sampling frequency of LAMMPS simulation')

    no_pbc: Boolean = Field(
        default=False,
        description='Whether to use periodic boundary condition')

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    resource: Dict[String, String] = Field(
        default={
            'image_name': 'registry.dp.tech/dptech/dpmd:2.2.8-cuda11.8',
            'scass_type': 'c8_m32_1 * NVIDIA V100',
        },
        description='Bohrium resource for LAMMPS exploration, no need to change in most cases',
    )

    cmd: String = Field(
        default='lmp',
        description="Command to run LAMMPS, note that it depends on the docker image you used")


class ModelDeviation(BaseModel):
    lo: Float = Field(
        default=0.1,
        description="Lower bound of the deviation")

    hi: Float = Field(
        default=0.5,
        description="Upper bound of the deviation")

    metric: String = Field(
        default='max_devi_f',
        description="Metric to measure the deviation")


class Cp2kSettings(BaseModel):
    input_template: InputFilePath = Field(
        description="Input template file for CP2K simulation")

    limit: Int = Field(
        default=50,
        description="Limit of the number of structures to be labeled for each iteration")

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    resource: Dict[String, String] = Field(
        default={
            'image_name': 'registry.dp.tech/dptech/cp2k:11',
            'scass_type': 'c32_m64_cpu',
        },
        description='Bohrium resource for CP2K simulation, no need to change in most cases',
    )

    cmd: String = Field(
        default='mpirun -np 32 cp2k.popt',
        description="Script to run CP2K simulation, note that it depends on the docker image")


class DynacatTeslaArgs(DFlowOptions):

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    max_iters: Int = Field(
        default = 7,
        description="Maximum iterations of the workflow")

    s3_prefix: Optional[String] = Field(
        description="Specify the S3 prefix of DFlow. By default a random prefix will be used for different jobs. Jobs use the same prefix will share the same working directory, which allow you to inherit the state of previous run.")

    deepmd: DeepmdSettings = Field(
        description="DeepMD settings for training")

    lammps: LammpsSetting = Field(
        description="LAMMPS settings for structure exploration")

    model_deviation: ModelDeviation = Field(
        description="Model deviation settings for screening structures")

    cp2k: Cp2kSettings = Field(
        description="CP2K settings for DFT labeling")

    output_dir : OutputDirectory = Field(
        default='./output',
        description="Output directory of LAMMPS simulation")

    python_resource: Dict[String, String] = Field(
        default={
            'image_name': 'registry.dp.tech/dptech/prod-13325/dflow-galaxy:0.1.4-main-8bb98c7',
            'scass_type': 'c2_m4_cpu',
        },
        description='Bohrium resource for Python scripts, no need to change in most cases',
    )


def launch_app(args: DynacatTeslaArgs) -> int:
    s3_prefix = args.s3_prefix or f'dynacat-{uuid4()}'
    logger.info(f'using s3 prefix: {s3_prefix}')
    tesla_template = get_res_path() / 'dynacat' / 'tesla_template.yml'

    shutil.copy(tesla_template, 'tesla-preset.yml')
    executor_config = _get_executor_config(args)
    workflow_config = _get_workflow_config(args)

    # TODO: use yaml to dump config pretty
    dump_json(executor_config, 'tesla-executor.yml')
    dump_json(workflow_config, 'tesla-workflow.yml')

    # copy generated configuration to output
    os.makedirs(args.output_dir, exist_ok=True)
    os.system(f'cp tesla-*.yml {args.output_dir}')

    if args.dry_run:
        logger.info('Skip running workflow due to dry_run is set to True')
        return 0

    setup_dflow_context(args)
    workflow = build_tesla_workflow('tesla-preset.yml', 'tesla-executor.yml', 'tesla-workflow.yml',
                                    s3_prefix=s3_prefix, max_iters=args.max_iters)
    try:
        workflow.run()
    finally:
        # reclaim useful data
        workflow.s3_download('iter-dataset', args.output_dir)
        workflow.s3_download('train-deepmd', args.output_dir)
    return 0


def _get_executor_config(args: DynacatTeslaArgs):
    return {
        'executors': {
            'bohrium': {
                'bohrium': {},
                'apps': {
                    'python': {
                        'resource': {
                            'bohrium': args.python_resource,
                        }
                    },
                    'deepmd': {
                        'resource': {
                            'bohrium': args.deepmd.resource,
                        },
                        'dp_cmd': args.deepmd.cmd,
                        'concurrency': args.deepmd.concurrency,

                    },
                    'lammps': {
                        'resource': {
                            'bohrium': args.lammps.resource,
                        },
                        'lammps_cmd': args.lammps.cmd,
                        'concurrency': args.lammps.concurrency,
                    },
                    'cp2k': {
                        'resource': {
                            'bohrium': args.cp2k.resource,
                        },
                        'cp2k_cmd': args.cp2k.cmd,
                        'concurrency': args.cp2k.concurrency,
                    }
                }
            }
        },
        'orchestration': {
            'deepmd': 'bohrium',
            'lammps': 'bohrium',
            'model_devi': 'bohrium',
            'cp2k': 'bohrium'
        },
    }


def _get_workflow_config(args: DynacatTeslaArgs):
    # process system file
    explore_data_file = args.lammps.system_file.get_full_path()
    atoms = ase.io.read(explore_data_file, index=0)
    type_map, mass_map = ai2cat.get_type_map(atoms)  # type: ignore

    cp2k_input_template = load_text(args.cp2k.input_template.get_full_path())
    deepmd_template = load_json(args.deepmd.input_template.get_full_path())
    product_vars, broadcast_vars = _get_lammps_vars(args.lammps.explore_vars)

    return {
        'datasets': {
            'train-data': {
                'url': args.deepmd.dataset.get_full_path(),
                'format': 'deepmd/npy',
            },
            'explore-data': {
                'url': explore_data_file,
            },
        },
        'workflow': {
            'general': {
                'type_map': type_map,
                'mass_map': mass_map,
            },
            'train': {
                'deepmd': {
                    'init_dataset': ['train-data'],
                    'input_template': deepmd_template,
                    'compress_model': args.deepmd.compress_model,
                }
            },
            'explore':{
                'lammps': {
                    'systems': ['explore-data'],
                    'nsteps': args.lammps.nsteps,
                    'ensemble': args.lammps.ensemble.value,
                    'timestep': args.lammps.timestep,
                    'sample_freq': args.lammps.sample_freq,
                    'no_pbc': args.lammps.no_pbc,
                    'plumed_config': args.lammps.plumed_config,
                    'product_vars': product_vars,
                    'broadcast_vars': broadcast_vars,
                    'template_vars': dict((item.key, item.value) for item in args.lammps.template_vars),
                }
            },
            'screen':{
                'model_devi': {
                    'metric': args.model_deviation.metric,
                    'decent_range': [args.model_deviation.lo, args.model_deviation.hi],
                }
            },
            'label': {
                'cp2k': {
                    'input_template': cp2k_input_template,
                    'limit': args.cp2k.limit,
                }
            },
        }
    }


def _get_lammps_vars(explore_vars: List[ExploreItem]):
    broadcast_vars = {}
    product_vars = {}
    for item in explore_vars:
        if item.broadcast:
            broadcast_vars[item.key] = _parse_string_array(item.value, dtype=float)
        else:
            product_vars[item.key] = _parse_string_array(item.value, dtype=float)
    return product_vars, broadcast_vars


def _parse_string_array(s: str, dtype=float, delimiter=','):
    return [dtype(x) for x in s.split(delimiter)]


def main():
    to_runner(
        DynacatTeslaArgs,
        launch_app,
        version="0.1.0",
    )(sys.argv[1:])


if __name__ == "__main__":
    main()
