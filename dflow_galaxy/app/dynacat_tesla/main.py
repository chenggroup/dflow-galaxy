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


class DynacatTeslaArgs(DFlowOptions):

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")


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
