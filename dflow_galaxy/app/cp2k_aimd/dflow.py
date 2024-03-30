import sys
import os
import glob

from dataclasses import dataclass
from dflow_galaxy.core import dflow, types, dispatcher
from dflow_galaxy.core.util import ensure_dirname, ensure_dir


@dataclass(frozen=True)
class RunCp2kTaskArgs:
    input_dir : types.InputArtifact
    output_dir: types.OutputArtifact


class RunCp2TaskFn:

    def __init__(self, cp2k_cmd: str, task_dir: str):
        self.cp2k_cmd = cp2k_cmd
        self.task_dir = task_dir

    def __call__(self, args: RunCp2kTaskArgs):
        """
        bash step to run cp2k aimd task
        """
        script = [
            f'cd {args.input_dir}/{self.task_dir}',
            f'{self.cp2k_cmd} -i cp2k.inp > cp2k.out',
            f'mkdir -p {args.output_dir}',
            f'mv * {args.output_dir}',
        ]
        return script

