from dataclasses import dataclass
from dflow_galaxy.core import dflow, types, dispatcher
import shutil


@dataclass(frozen=True)
class RunLammpsArgs:
    input_dir : types.InputArtifact
    output_dir: types.OutputArtifact


class RunLammpsFn:
    def __init__(self, lammps_script: str):
        self.lammps_script = lammps_script

    def __call__(self, args: RunLammpsArgs):
        script = [
            f'cd {args.input_dir}',
            self.lammps_script,
            f'mkdir -p {args.output_dir}',
            f'mv * {args.output_dir}',
        ]
        return script


def run_lammps_workflow(input_dir: str,
                        out_dir: str,
                        lammps_image: str,
                        lammps_device_model: str,
                        lammps_script: str):

    # bohrium dispatcher will be configured in bohrium.config
    # so here we just leave it empty
    bohrium_config = dispatcher.BohriumConfig()

    # start to build workflow
    dflow_builder = dflow.DFlowBuilder('lammps', s3_prefix='lammps')

    # setup and add lammps step to workflow
    lammps_res = dispatcher.Resource(
        bohrium=dispatcher.BohriumInputData(
            image_name=lammps_image,
            scass_type=lammps_device_model,
            disk_size=100,
        )
    )

    dflow_builder.s3_upload(input_dir, 'lammps_input')
    lammps_executor = dispatcher.create_bohrium_dispatcher(bohrium_config, lammps_res)
    lammps_fn = RunLammpsFn(lammps_script=lammps_script)
    lammps_step = dflow_builder.make_bash_step(lammps_fn, executor=lammps_executor)(
        RunLammpsArgs(input_dir='s3://./lammps_input', output_dir='s3://./lammps_output')
    )
    dflow_builder.add_step(lammps_step)

    # run workflow
    dflow_builder.run()

    # download artifacts to out_dir
    dflow_builder.s3_download('lammps_output')
    shutil.move('lammps_output', f'{out_dir}/lammps_output.tgz')
