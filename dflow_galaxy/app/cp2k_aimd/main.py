from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath
from dp.launching.typing import Int, String, Enum, Float
from dp.launching.cli import to_runner, default_minimal_exception_handler


class SystemTypeOptions(String, Enum):
    metal = "metal"
    semi = 'semi'


class AccuracyOptions(String, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class BasicSetOptions(String, Enum):
    ...

class PotentialOptions(String, Enum):
    ...


class InputArgs(BaseModel):
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

    raw_cp2k_input: InputFilePath = Field(
        description='Advance option, provide your own CP2K input file, note that if you provide this file, the above settings will be ignored.')


def main():
    ...
