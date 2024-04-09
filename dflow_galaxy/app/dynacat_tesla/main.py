from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath
from dp.launching.typing import Int, String, Enum, Float, Boolean, List, Optional
from dp.launching.cli import to_runner, default_minimal_exception_handler

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context, EnsembleOptions
from dflow_galaxy.res import get_res_path
from dflow_galaxy.core.log import get_logger

from dflow_galaxy.workflow.tesla.main import run_tesla, TeslaConfig

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

    image: String = Field(
        default='registry.dp.tech/dptech/dpmd:2.2.8-cuda11.8',
        description="Docker image for running DeepMD training")

    device_model: String = Field(
        default='c8_m32_1 * NVIDIA V100',
        description="Device model for DeepMD training")

    cmd: String = Field(
        default='dp',
        description="Command to run DeepMD, note that it depends on the docker image you used")


class LammpsSetting(BaseModel):
    systems: InputFilePath = Field(
        description="Structure file in extxyz or POSCAR format use for LAMMPS simulation")

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
        description="Template variables for LAMMPS simulation")

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    image: String = Field(
        default='registry.dp.tech/dptech/dpmd:2.2.8-cuda11.8',
        description="Docker image for running LAMMPS exploration")

    device_model: String = Field(
        default='c8_m32_1 * NVIDIA V100',
        description="Device model for LAMMPS exploration")

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

    image: String = Field(
        default='registry.dp.tech/dptech/cp2k:11',
        description="Docker image for running CP2K simulation")

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    device_model: String = Field(
        default='c32_m64_cpu',
        description="Device model for CP2K simulation")

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


def launch_app(args: DynacatTeslaArgs) -> int:
    s3_prefix = args.s3_prefix or f'dynacat-{uuid4()}'
    logger.info(f'using s3 prefix: {s3_prefix}')
    tesla_template = get_res_path() / 'dynacat' / 'tesla_template.yml'
    deepmd_template = get_res_path() / 'dynacat' / 'deepmd_template.yml'

    setup_dflow_context(args)


    return 0


def main():
    to_runner(
        DynacatTeslaArgs,
        launch_app,
        version="0.1.0",
    )(sys.argv[1:])


if __name__ == "__main__":
    main()
