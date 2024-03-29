from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath
from dp.launching.typing import Int, String, Enum, Float, Boolean
from dp.launching.cli import to_runner, default_minimal_exception_handler


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


class InputArgs(BaseModel):
    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    system_file: InputFilePath = Field(
        description="Upload a system file as the initial structure of AIMD simulation")

    system_type: SystemTypeOptions = Field(
        default=SystemTypeOptions.metal,
        description="Type of the system")

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

    raw_cp2k_input: InputFilePath = Field(
        description='Provide your own CP2K input file, note that the above settings will be ignored.')


def main_entry(args: InputArgs) -> int:
    status = 0
    from pprint import pprint
    pprint(args.dict())
    return status


def to_parser():
    return to_runner(
        InputArgs,
        main_entry,
        version="0.1.0",
        exception_handler=default_minimal_exception_handler
    )

if __name__ == "__main__":
    import sys
    to_parser()(sys.argv[1:])
