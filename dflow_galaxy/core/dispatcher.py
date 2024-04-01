from .pydantic import BaseModel

from typing import Optional
from urllib.parse import urlparse
import os


class BohriumInputData(BaseModel):
    image_name: str
    job_type: str = 'container'
    platform: str = 'ali'
    scass_type: str = 'c2_m4_cpu'
    job_name: str = 'bohrium_job'
    disk_size: int = 20  # in GB


class Resource(BaseModel):
    queue: Optional[str] = None
    image: Optional[str] = None
    sub_path: str = '.'

    nodes: int = 1
    cpu_per_node: int = 1
    gpu_per_node: int = 0

    # use this for job running on bohrium
    bohrium: Optional[BohriumInputData] = None


    def get_resource_dict(self):
        return {
            'number_node': self.nodes,
            'cpu_per_node': self.cpu_per_node,
            'gpu_per_node': self.gpu_per_node,
        }


class BohriumConfig(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    project_id: Optional[str] = None


class HpcConfig(BaseModel):
    class SlurmConfig(BaseModel):
        ...
    class LsfConfig(BaseModel):
        ...
    class PBSConfig(BaseModel):
        ...

    url: str
    """
    SSH URL to connect to the HPC, for example: `john@hpc-login01`
    """
    key_file: Optional[str] = None
    """
    Path to the private key file for SSH connection
    """
    slurm: Optional[SlurmConfig] = None
    lsf: Optional[LsfConfig] = None
    pbs: Optional[PBSConfig] = None
    base_dir: str = '.'
    clean: bool = False

    def get_context_type(self):
        if self.slurm:
            return 'Slurm'
        if self.lsf:
            return 'LSF'
        if self.pbs:
            return 'PBS'
        raise ValueError('At least one of slurm, lsf or pbs should be provided')


class ExecutorConfig(BaseModel):
    hpc: Optional[HpcConfig] = None
    bohrium: Optional[BohriumConfig] = None


def create_dispatcher(config: ExecutorConfig, resource: Resource):
    """
    Create a dispatcher executor based on the configuration
    """
    if config.hpc:
        return create_hpc_dispatcher(config.hpc, resource)
    elif config.bohrium:
        return create_bohrium_dispatcher(config.bohrium, resource)
    raise ValueError('At least one of hpc or bohrium should be provided')


def create_bohrium_dispatcher(config: BohriumConfig, resource: Resource):
    if resource.bohrium is None:
        raise ValueError('Bohrium resource is required when using Bohrium dispatcher')
    from dflow.plugins.dispatcher import DispatcherExecutor
    password = os.environ.get('BOHRIUM_PASSWORD') or config.password

    # remote profile can also be config via global bohrium.config
    remote_profile = {}
    if config.email:
        remote_profile['email'] = config.email

    if config.project_id:
        # I don't know which one is correct, so I set both
        remote_profile['program_id'] = config.project_id
        remote_profile['project_id'] = config.project_id
    if password:
        remote_profile['password'] = password

    remote_profile['input_data'] = resource.bohrium.dict()
    machine_dict = {
        'batch_type': 'Bohrium',
        'context_type': 'Bohrium',
        'remote_profile': remote_profile,
    }
    return DispatcherExecutor(
        machine_dict=machine_dict,
        # the following are not used in Bohrium dispatcher
        image=resource.image,
        resources_dict= resource.get_resource_dict(),
    )


def create_hpc_dispatcher(config: HpcConfig, resource: Resource):
    from dflow.plugins.dispatcher import DispatcherExecutor
    url = urlparse(config.url)
    assert url.scheme == 'ssh', 'Only SSH is supported for HPC dispatcher'
    assert url.username, 'Username is required in the URL'
    assert os.path.isabs(config.base_dir), 'Base directory must be an absolute path'
    remote_root = os.path.normpath(
        os.path.join(config.base_dir, resource.sub_path))

    remote_profile = { }
    if config.key_file:
        remote_profile['key_filename'] = config.key_file
    machine_dict = {
        "batch_type": config.get_context_type(),
        "context_type": "SSHContext",
        'remote_profile': remote_profile,
    }

    return DispatcherExecutor(
        host=url.hostname or 'localhost',
        private_key_file=config.key_file,  # type: ignore
        username=url.username,
        port=url.port or 22,
        clean=config.clean,
        machine_dict=machine_dict,
        resources_dict=resource.get_resource_dict(),
        queue_name=resource.queue,
        remote_root=remote_root,
    )


class BaseApp(BaseModel):
    resource: Resource
    setup_script: str = ''


class PythonApp(BaseApp):
    python_cmd: str = 'python3'
    max_worker: int = 4
