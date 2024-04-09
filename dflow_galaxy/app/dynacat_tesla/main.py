from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath
from dp.launching.typing import Int, String, Enum, Float, Boolean, List, Optional, Dict
from dp.launching.cli import to_runner, default_minimal_exception_handler

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context, EnsembleOptions
from dflow_galaxy.res import get_res_path
from dflow_galaxy.core.log import get_logger

from dflow_galaxy.workflow.tesla.main import run_tesla, TeslaConfig

from ai2_kit.core.util import dump_json, load_text
from ai2_kit.feat import catalysis as ai2cat

from pathlib import Path
from uuid import uuid4
import shutil
import sys


logger = get_logger(__name__)


class KvItem(BaseModel):
    key: String = Field(
        description="Key of the item")
    value: String = Field(
        format='multi-line',
        description="Value of the item")


class DeepmdSettings(BaseModel):
    dataset : InputFilePath = Field(
        description="DeepMD dataset folder in npy format")

    input_template: Optional[InputFilePath] = Field(
        description="Input template file for DeepMD training, use build-in template if not provided")

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
    systems: InputFilePath = Field(
        description="Structure file in extxyz or POSCAR format use for LAMMPS simulation")

    ensemble: EnsembleOptions = Field(
        default=EnsembleOptions.csvr,
        description='Ensemble of LAMMPS simulation')

    plumed_config: String = Field(
        description='Plumed configuration file for metadynamics simulation')

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
    if args.deepmd.input_template:
        deepmd_template = args.deepmd.input_template.get_full_path()
    else:
        deepmd_template = get_res_path() / 'dynacat' / 'deepmd_template.json'

    shutil.copy(tesla_template, 'tesla-preset.yml')
    shutil.copy(deepmd_template, 'deepmd.json')

    executor_config = _get_executor_config(args)
    workflow_config = _get_workflow_config(args)

    dump_json(executor_config, 'tesla-executor.yml')
    dump_json(workflow_config, 'tesla-workflow.yml')


    setup_dflow_context(args)


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
    explore_data = args.lammps.systems.get_full_path()
    cp2k_input_template = load_text(args.cp2k.input_template.get_full_path())

    return {
        'datasets': {
            'train-data': {
                'url': args.deepmd.dataset.get_full_path(),
                'format': 'deepmd/npy',
            },
            'explore-data': {
                'url': explore_data,
            },
        },
        'workflow': {
            'general': {
                'type_map': [],
                'mass_map': [],
            },
            'train': {
                'deepmd': {

                }
            },
            'explore':{
                'lammps': {

                }
            },
            'screen':{
                'model_devi': {

                }
            },
            'label': {
                'cp2k': {
                    'input_template': cp2k_input_template,
                }
            },
        }
    }



def main():
    to_runner(
        DynacatTeslaArgs,
        launch_app,
        version="0.1.0",
    )(sys.argv[1:])


if __name__ == "__main__":
    main()
