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

from .dflow import run_cp2k_workflow

logger = get_logger(__name__)


def get_cp2k_data_file(name: str):
    data_file =  get_cp2k_data_dir() / name
    return str(data_file)


class SystemTypeOptions(String, Enum):
    metal = "metal"
    semi = 'semi'


class AccuracyOptions(String, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class BasicSetOptions(String, Enum):
    HFX_BASIS = "HFX_BASIS"
    BASIS_ADMM = "BASIS_ADMM"
    BASIS_def2_QZVP_RI_ALL = "BASIS_def2_QZVP_RI_ALL"
    BASIS_MOLOPT_LnPP1 = "BASIS_MOLOPT_LnPP1"
    BASIS_MOLOPT_AcPP1 = "BASIS_MOLOPT_AcPP1"
    BASIS_SET = "BASIS_SET"
    BASIS_MOLOPT_UZH = "BASIS_MOLOPT_UZH"
    BASIS_ADMM_ae = "BASIS_ADMM_ae"
    ALL_BASIS_SETS = "ALL_BASIS_SETS"
    BASIS_LRIGPW_AUXMOLOPT = "BASIS_LRIGPW_AUXMOLOPT"
    BASIS_MOLOPT = "BASIS_MOLOPT"
    BASIS_MINIX = "BASIS_MINIX"
    BASIS_MOLOPT_UCL = "BASIS_MOLOPT_UCL"
    BASIS_ZIJLSTRA = "BASIS_ZIJLSTRA"
    GTH_BASIS_SETS = "GTH_BASIS_SETS"
    EMSL_BASIS_SETS = "EMSL_BASIS_SETS"
    BASIS_MINBAS = "BASIS_MINBAS"
    BASIS_ccGRB_UZH = "BASIS_ccGRB_UZH"
    BASIS_ADMM_MOLOPT = "BASIS_ADMM_MOLOPT"
    BASIS_ADMM_UZH = "BASIS_ADMM_UZH"
    BASIS_pob = "BASIS_pob"
    BASIS_PERIODIC_GW = "BASIS_PERIODIC_GW"
    BASIS_RI_cc_TZ = "BASIS_RI_cc-TZ"
    BASIS_MOLOPT_LnPP2 = "BASIS_MOLOPT_LnPP2"


class PotentialOptions(String, Enum):
    GTH_SOC_POTENTIALS = "GTH_SOC_POTENTIALS"
    LnPP2_POTENTIALS = "LnPP2_POTENTIALS"
    HF_POTENTIALS = "HF_POTENTIALS"
    GTH_POTENTIALS = "GTH_POTENTIALS"
    ALL_POTENTIALS = "ALL_POTENTIALS"
    ECP_POTENTIALS_pob_TZVP_rev2 = "ECP_POTENTIALS_pob-TZVP-rev2"
    ECP_POTENTIALS = "ECP_POTENTIALS"
    NLCC_POTENTIALS = "NLCC_POTENTIALS"
    AcPP1_POTENTIALS = "AcPP1_POTENTIALS"
    LnPP1_POTENTIALS = "LnPP1_POTENTIALS"


class Cp2kLightningArgs(DFlowOptions):

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    system_file: InputFilePath = Field(
        description="A system file as the initial structure of AIMD simulation")

    system_type: SystemTypeOptions = Field(
        default=SystemTypeOptions.metal,
        description="Type of the system, semi for semi-conductor, metal for metal")

    accuracy: AccuracyOptions = Field(
        default=AccuracyOptions.medium,
        description="Accuracy of the simulation, the higher the accuracy, the longer the simulation time")

    temperature: Float = Field(
        default=300.0,
        description="Temperature of the simulation in K")

    steps: Int = Field(
        default=1000,
        description="Number of steps in the simulation")

    timestep: Float = Field(
        default=0.5,
        description="Time step of the simulation in fs")

    basis_set: BasicSetOptions = Field(
        default=BasicSetOptions.BASIS_MOLOPT,
        description='Select the basis set for the simulation')

    potential: PotentialOptions = Field(
        default=PotentialOptions.GTH_POTENTIALS,
        description='Select the potential for the simulation')

    output_dir: OutputDirectory = Field(
        default="./output",
        description="Output directory for the simulation results")

    cp2k_image: String = Field(
        default='registry.dp.tech/dptech/cp2k:11',
        description="Docker image for running CP2K simulation")

    cp2k_device_model: String = Field(
        default='c32_m64_cpu',
        description="Device model for CP2K simulation")

    cp2k_script: String = Field(
        default= '\n'.join([
            '# 1. the output file must be named as `output`',
            '# 2. cp2k.aimd.inp for aimd and cp2k.dft.inp for dft',
            '# Note that different container may have different setup',
            'source /opt/cp2k-toolchain/install/setup',
            'mpirun -n 32 cp2k.popt -i cp2k.aimd.inp &> output',
            'rm *.wfn  # reduce the size of output file',
        ]),
        format='multi-line',
        description="Script to run CP2K simulation, note that it depends on the docker image")


def launch_app(args: Cp2kLightningArgs) -> int:
    # stage 1: generate cp2k input file
    basis_set_file = get_cp2k_data_file(args.basis_set.value)
    potential_file = get_cp2k_data_file(args.potential.value)
    # copy data file to cwd
    # don't use absolute path as the config file will be use in docker
    shutil.copy(basis_set_file, '.')
    shutil.copy(potential_file, '.')

    # create a closure to generate cp2k input file
    def _gen_cp2k_input(out_dir: str, aimd: bool):
        config_builder = ai2cat.ConfigBuilder()
        config_builder.load_system(args.system_file.get_full_path()).gen_cp2k_input(
            out_dir=out_dir,
            aimd=aimd,
            # common options
            style=args.system_type.value,
            temp=args.temperature,
            steps=args.steps,
            timestep=args.timestep,
            basic_set_file=args.basis_set.value,
            potential_file=args.potential.value,
            parameter_file='dftd3.dat',
        )
    _gen_cp2k_input('aimd', aimd=True)
    _gen_cp2k_input('dft', aimd=False)

    out_dir = Path(args.output_dir)
    shutil.move('aimd/cp2k.inp', out_dir / 'cp2k.aimd.inp')
    shutil.move('dft/cp2k.inp', out_dir / 'cp2k.dft.inp')
    shutil.move('aimd/coord_n_cell.inc', out_dir / 'coord_n_cell.inc')


    if args.dry_run:
        logger.info('skip dflow run due to dry_run is set to True')
        return 0

    # stage 2: run cp2k with dflow
    setup_dflow_context(args)
    run_cp2k_workflow(
        input_dir=str(args.output_dir),
        out_dir=str(args.output_dir),
        cp2k_device_model=str(args.cp2k_device_model),
        cp2k_image=str(args.cp2k_image),
        cp2k_script=str(args.cp2k_script),
    )
    return 0

def main():
    to_runner(
        Cp2kLightningArgs,
        launch_app,
        version="0.1.0",
    )(sys.argv[1:])


if __name__ == "__main__":
    main()
