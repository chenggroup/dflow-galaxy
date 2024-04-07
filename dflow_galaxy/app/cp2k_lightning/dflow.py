from dataclasses import dataclass
from dflow_galaxy.core import dflow, types, dispatcher
import shutil


@dataclass(frozen=True)
class RunCp2kArgs:
    input_dir : types.InputArtifact
    output_dir: types.OutputArtifact


class RunCp2kFn:
    def __init__(self, cp2k_script: str):
        self.cp2k_script = cp2k_script

    def __call__(self, args: RunCp2kArgs):
        """
        bash step to run cp2k aimd task
        """
        script = [
            # guess cp2k data dir
            '[[ -z "${CP2K_DATA_DIR}" ]] && export CP2K_DATA_DIR="$(dirname "$(which cp2k)")/../../data" || true',
            f'cd {args.input_dir}',
            self.cp2k_script,
            f'mkdir -p {args.output_dir}',
            f'mv * {args.output_dir}',
        ]
        return script


def run_cp2k_workflow(input_dir: str,
                      out_dir: str,
                      cp2k_image: str,
                      cp2k_device_model: str,
                      cp2k_script: str):

    # bohrium dispatcher will be configured in bohrium.config
    # so here we just leave it empty
    bohrium_config = dispatcher.BohriumConfig()

    # start to build workflow
    dflow_builder = dflow.DFlowBuilder('cp2k', s3_prefix='cp2k')

    # setup and add cp2k step to workflow
    cp2k_res = dispatcher.Resource(
        bohrium=dispatcher.BohriumInputData(
            image_name=cp2k_image,
            scass_type=cp2k_device_model,
            disk_size=100,
        )
    )

    dflow_builder.s3_upload(input_dir, 'cp2k_input')
    cp2k_executor = dispatcher.create_bohrium_dispatcher(bohrium_config, cp2k_res)
    cp2k_fn = RunCp2kFn(cp2k_script=cp2k_script)
    cp2k_step = dflow_builder.make_bash_step(cp2k_fn, executor=cp2k_executor)(
        RunCp2kArgs(input_dir='s3://./cp2k_input', output_dir='s3://./cp2k_output')
    )
    dflow_builder.add_step(cp2k_step)

    # run workflow
    dflow_builder.run()

    # download artifacts to out_dir
    dflow_builder.s3_download('cp2k_output')
    shutil.move('cp2k_output', f'{out_dir}/cp2k_output.tgz')
