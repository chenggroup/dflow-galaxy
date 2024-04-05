from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath, Optional
from dp.launching.typing import Int, String, Enum, Float, Boolean
from dp.launching.cli import to_runner, default_minimal_exception_handler

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context
from dflow_galaxy.res import get_cp2k_data_dir
from dflow_galaxy.core.log import get_logger
from ai2_kit.feat import catalysis as ai2cat

from pathlib import Path
import shutil
import sys


logger = get_logger(__name__)


class DynaCatMdArgs(DFlowOptions):

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    system_file: InputFilePath = Field(
        description="A system file as the initial structure of LAMMPS simulation")

    steps: Int = Field(
        default=20000,
        description='Steps of LAMMPS simulation')

    step_size: Float = Field(
        default=0.0005,
        description='Time step size of LAMMPS simulation')

    temperature: Float = Field(
        default=330,
        description='Temperature of LAMMPS simulation')

    plumed_config: String = Field(
        format='multi-line',
        description='Plumed configuration file for metadynamics simulation')


def launch_app(args: DynaCatMdArgs) -> int:
    return 0


def main():
    to_runner(
        DynaCatMdArgs,
        launch_app,
        version="0.1.0",
        exception_handler=default_minimal_exception_handler
    )(sys.argv[1:])


if __name__ == '__main__':
    main()
