from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath, Optional
from dp.launching.typing import Int, String, Enum, Float, Boolean
from dp.launching.cli import to_runner, default_minimal_exception_handler

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context, EnsembleOptions
from dflow_galaxy.res import get_cp2k_data_dir
from dflow_galaxy.core.log import get_logger
from ai2_kit.feat import catalysis as ai2cat

from pathlib import Path
import shutil
import sys


logger = get_logger(__name__)


class DeepmdSettings(BaseModel):
    dataset : InputFilePath = Field(
        description="DeepMD dataset folder in npy format")

    dataset_limit: Int = Field(
        default=0,
        description="If the dataset is too large, you can use this parameter to limit the size of the dataset, 0 means no limit")

    image: String = Field(
        default='registry.dp.tech/dptech/dpmd:2.2.8-cuda11.8',
        description="Docker image for running DeepMD training")

    device_model: String = Field(
        default='c8_m32_1 * NVIDIA V100',
        description="Device model for DeepMD training")

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    cmd: String = Field(
        default='dp',
        description="Command to run DeepMD, note that it depends on the docker image you used")


class LammpsSetting(BaseModel):
    systems: InputFilePath = Field(
        description="Structure file in xyz format use for LAMMPS simulation")

    systems_limit: Int = Field(
        default=0,
        description="If the system file is too large, you can use this parameter to limit the structure in systems, 0 means no limit")


class Cp2kSettings(BaseModel):
    input_template: InputFilePath = Field(
        description="Input template file for CP2K simulation")

    image: String = Field(
        default='registry.dp.tech/dptech/cp2k:11',
        description="Docker image for running CP2K simulation")

    device_model: String = Field(
        default='c32_m64_cpu',
        description="Device model for CP2K simulation")

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    cmd: String = Field(
        default='mpirun -np 32 cp2k.popt',
        description="Script to run CP2K simulation, note that it depends on the docker image")


class DynacatTeslaArgs(DFlowOptions):

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    deepmd: DeepmdSettings = Field(
        description="DeepMD settings for training")

    lammps: LammpsSetting = Field(
        description="LAMMPS settings for structure exploration")

    cp2k: Cp2kSettings = Field(
        description="CP2K settings for DFT labeling")





def launch_app(args: DynacatTeslaArgs) -> int:
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
