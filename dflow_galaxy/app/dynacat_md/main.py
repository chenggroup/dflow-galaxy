from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath, Optional, Dict
from dp.launching.typing import Int, String, Enum, Float, Boolean
from dp.launching.cli import to_runner, default_minimal_exception_handler

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context
from dflow_galaxy.core.log import get_logger
from ai2_kit.feat import catalysis as ai2cat
from ai2_kit.core.util import dump_text, dump_json


from pathlib import Path
import shutil
import sys
import os

from .dflow import run_lammps_workflow

logger = get_logger(__name__)


class EnsembleOptions(String, Enum):
    nvt = 'nvt'
    nvt_i = 'nvt-i'
    nvt_a = 'nvt-a'
    nvt_iso = 'nvt-iso'
    nvt_aniso = 'nvt-aniso'
    npt = 'npt'
    npt_t = 'npt-t'
    npt_tri = 'npt-tri'
    nve = 'nve'
    csvr = 'csvr'


class DynaCatMdArgs(DFlowOptions):

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    system_file: InputFilePath = Field(
        description="A system file as the initial structure of LAMMPS simulation")

    # TODO: support multiple deepmd models
    deepmd_model: InputFilePath = Field(
        description="Deepmd model file for LAMMPS simulation")

    ensemble: EnsembleOptions = Field(
        default=EnsembleOptions.csvr,
        description='Ensemble of LAMMPS simulation')

    temperature: Float = Field(
        default=330,
        description='Temperature of LAMMPS simulation')

    plumed_config: String = Field(
        format='multi-line',
        description='Plumed configuration file for metadynamics simulation')

    pressure: Float = Field(
        default=-1,
        description='Pressure of LAMMPS simulation, should be -1 unless it is NPT ensemble')

    steps: Int = Field(
        default=20000,
        description='Steps of LAMMPS simulation')

    step_size: Float = Field(
        default=0.0005,
        description='Time step size of LAMMPS simulation in ps')

    sample_freq: Int = Field(
        title='Sampling Frequency',
        default=10,
        description='Sampling frequency of LAMMPS simulation')

    extra_args: Dict[String, Float] = Field(
        title='Extra Arguments',
        default={
            'tau_t': 0.1,
            'tau_p': 0.5,
            'time_const': 0.1,
        },
        description="Extra arguments for LAMMPS simulation, e.g. tau_t, tau_p, time_const, etc. Don't remove or add extra arguments if you are not sure about it."
    )

    output_dir : OutputDirectory = Field(
        default='./output',
        description="Output directory of LAMMPS simulation")

    lammps_image: String = Field(
        default='registry.dp.tech/dptech/dpmd:2.2.8-cuda11.8',
        description="Docker image for running LAMMPS simulation")

    lammps_device_model: String = Field(
        default='c8_m32_1 * NVIDIA V100',
        description="Device model for LAMMPS simulation")

    lammps_script: String = Field(
        default= '\n'.join([
            '# LAMMPS input can be referenced as lammps.inp',
            '# Note that different container may have different setup',
            'lmp -in lammps.inp &> lammps.out',
        ]),
        format='multi-line',
        description="Script to run LAMMPS simulation, note that it depends on the docker image")


def launch_app(args: DynaCatMdArgs) -> int:
    config_builder = ai2cat.ConfigBuilder()

    shutil.copy(args.deepmd_model, 'dp-model.pb')
    dump_text(args.plumed_config, 'plumed.inp')

    logger.info(f'type of system_file: {type(args.system_file)}')
    config_builder.load_system(args.system_file.get_full_path()).gen_lammps_input(
        out_dir=args.output_dir,
        nsteps=args.steps,
        temp=args.temperature,
        sample_freq=args.sample_freq,
        pres=args.pressure,
        abs_path=False,
        ensemble=args.ensemble,
        dp_models=['dp-model.pb'],
        **args.extra_args,
    )
    shutil.move('plumed.inp', args.output_dir)
    shutil.move('dp-model.pb', args.output_dir)

    if args.dry_run:
        return 0

    setup_dflow_context(args)
    run_lammps_workflow(
        input_dir=str(args.output_dir),
        out_dir=str(args.output_dir),
        lammps_image=str(args.lammps_image),
        lammps_device_model=str(args.lammps_device_model),
        lammps_script=str(args.lammps_script),
    )

    return 0


def main():
    to_runner(
        DynaCatMdArgs,
        launch_app,
        version="0.1.0",
    )(sys.argv[1:])


if __name__ == '__main__':
    main()
