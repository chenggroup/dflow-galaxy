from dflow.op_template import ScriptOPTemplate
from dflow.plugins.dispatcher import DispatcherExecutor
import dflow

from typing import Final, Callable, TypeVar, Optional, Union, Annotated, Generic, Dict, Any, Iterable
from dataclasses import fields, is_dataclass
from urllib.parse import urlparse
from collections import namedtuple
from pathlib import Path
from uuid import uuid4
import cloudpickle as cp
import tempfile
import tarfile
import hashlib
import inspect
import base64
import shutil
import shlex
import bz2
import os


from .util import resolve_ln
from .log import get_logger
from . import types

logger = get_logger(__name__)

T_ARGS = TypeVar('T_ARGS')
T_RESULT = TypeVar('T_RESULT')

DFLOW_ARTIFACT = Union[str, dflow.S3Artifact, dflow.OutputArtifact, dflow.InputArtifact]


class Step(Generic[T_ARGS, T_RESULT]):
    def __init__(self, df_step: dflow.Step):
        self.df_step = df_step

    @property
    def args(self) -> T_ARGS:
        return ObjProxy(self.df_step.inputs.parameters,
                        self.df_step.inputs.artifacts,
                        self.df_step.outputs.artifacts)  # type: ignore

    @property
    def result(self) -> T_RESULT:
        return ObjProxy(self.df_step.outputs.parameters)  # type: ignore

Steps = Union[Step, Iterable['Steps']]


class ObjProxy:
    def __init__(self, *obj):
        self.objs = obj

    def __getattr__(self, name):
        for obj in self.objs:
            if name in obj:
                return obj[name]
        raise AttributeError(f'{name} not found in {self.objs}')


def pickle_converts(obj, pickle_module='cp', bz2_module='bz2', base64_module='base64'):
    """
    convert an object to its pickle string form
    """
    obj_pkl = cp.dumps(obj, protocol=cp.DEFAULT_PROTOCOL)
    compress_level = 5 if len(obj_pkl) > 4096 else 1
    compressed = bz2.compress(obj_pkl, compress_level)
    obj_b64 = base64.b64encode(compressed).decode('ascii')
    return f'{pickle_module}.loads({bz2_module}.decompress({base64_module}.b64decode({repr(obj_b64)})))'


def iter_python_step_args(obj):
    """
    Iterate over the input fields of a python step.
    A python step input should be:
    1. A frozen dataclass.
    2. All fields are annotated with InputParam, InputArtifact or OutputArtifact.
    """
    assert is_dataclass(obj), f'{obj} is not a dataclass'
    assert obj.__dataclass_params__.frozen, f'{obj} is not frozen'
    for f in fields(obj):
        msg = f'{f.name} is not annotated with InputParam, InputArtifact or OutputArtifact'
        assert hasattr(f.type, '__metadata__'), msg
        assert f.type.__metadata__[0] in (
            types.Symbol.INPUT_PARAMETER, types.Symbol.INPUT_ARTIFACT,
            types.Symbol.OUTPUT_ARTIFACT), msg
        yield f, getattr(obj, f.name, None)


def iter_python_step_return(obj):
    """
    Iterate over the output fields of a python step.
    A python step output should be:
    1. A dataclass.
    2. All fields are annotated with OutputParam.
    """
    if obj is inspect.Signature.empty or obj is None:
        return

    assert is_dataclass(obj), f'{obj} is not a dataclass'
    for f in fields(obj):
        msg = f'{f.name} is not annotated with OutputParam'
        assert hasattr(f.type, '__metadata__'), msg
        assert f.type.__metadata__ [0] == types.Symbol.OUTPUT_PARAMETER, msg
        yield f, getattr(obj, f.name, None)


_BashTemplate = namedtuple('_BashStep', ['source',
                                         'dflow_input_parameters',
                                         'dflow_input_artifacts',
                                         'dflow_output_artifacts',
                                         ])


def bash_build_template(py_fn: Callable,
                        base_dir: str,
                        setup_script: str = '',
                        default_archive: Optional[str] = 'default',
                        eof: str = '__EOF__') -> _BashTemplate:
    """
    build bash step from a python function
    """
    sig = inspect.signature(py_fn)
    assert len(sig.parameters) == 1, f'{py_fn} should have only one parameter'
    args_type = sig.parameters[next(iter(sig.parameters))].annotation

    input_artifacts_dir = os.path.join(base_dir, 'input-artifacts')
    output_artifacts_dir = os.path.join(base_dir, 'output-artifacts')

    dflow_input_parameters: Dict[str, dflow.InputParameter] = {}
    dflow_input_artifacts: Dict[str, dflow.InputArtifact] = {}
    dflow_output_artifacts: Dict[str, dflow.OutputArtifact] = {}

    args_dict = {}

    source = [
        '#!/bin/bash',
        'set -e',
        'ARGO_PROGRESS_FILE=${ARGO_PROGRESS_FILE:-./argo_progress_file.txt}',
        setup_script,
        ''
        f'mkdir -p {shlex.quote(output_artifacts_dir)}',
        '',
        '# Setup Variables',
    ]

    for f, v in iter_python_step_args(args_type):
        meta = f.type.__metadata__[0]
        if meta == types.Symbol.INPUT_PARAMETER:
            # input parameter can be a multiline string
            bash_name = f'_DF_INPUT_PARAMETER_{f.name}_'
            source.extend([
                f"{bash_name}=$(cat << {eof}",
                f"{{{{inputs.parameters.{f.name}}}}}",
                eof,
                ')',
            ])
            dflow_input_parameters[f.name] = dflow.InputParameter(name=f.name)
            args_dict[f.name] = f'${bash_name}'

        elif meta == types.Symbol.INPUT_ARTIFACT or isinstance(meta, dflow.InputArtifact):
            bash_name = f'_DF_INPUT_ARTIFACT_{f.name}_'
            path = os.path.join(input_artifacts_dir, f.name)
            source.append(f'{bash_name}={shlex.quote(path)}')
            artifact = meta
            if not isinstance(artifact, dflow.InputArtifact):
                artifact = dflow.InputArtifact(path=path, archive=default_archive)  # type: ignore
            dflow_input_artifacts[f.name] = artifact
            args_dict[f.name] = f'${bash_name}'

        elif meta == types.Symbol.OUTPUT_ARTIFACT or isinstance(meta, dflow.OutputArtifact):
            bash_name = f'_DF_OUTPUT_ARTIFACT_{f.name}_'
            path = os.path.join(output_artifacts_dir, f.name)
            source.append(f'{bash_name}={shlex.quote(path)}')
            artifact = meta
            if not isinstance(artifact, dflow.OutputArtifact):
                artifact = dflow.OutputArtifact(path=path, archive=default_archive)  # type: ignore
            dflow_output_artifacts[f.name] = artifact

            args_dict[f.name] = f'${bash_name}'
        else:
            raise ValueError(f'unsupported type {f.type}')

    bash_script = py_fn(ObjProxy(args_dict))
    if isinstance(bash_script, list):
        bash_script = '\n'.join(bash_script)
    assert isinstance(bash_script, str), f'{py_fn} should return a string or a list of string'

    source.extend([
        '',
        '#' * 80,
        '',
        bash_script,
    ])

    _dflow_script_check(source, base_dir)
    return _BashTemplate(source='\n'.join(source),
                         dflow_input_parameters=dflow_input_parameters,
                         dflow_input_artifacts=dflow_input_artifacts,
                         dflow_output_artifacts=dflow_output_artifacts)


_PythonTemplate = namedtuple('_PythonStep', ['source', 'fn_str', 'script_path', 'pkg_dir',
                                             'dflow_input_parameters',
                                             'dflow_input_artifacts',
                                             'dflow_output_parameters',
                                             'dflow_output_artifacts',
                                             ])


def python_build_template(py_fn: Callable,
                          base_dir: str,
                          setup_script: str = '',
                          python_cmd: str = 'python3',
                          default_archive: Optional[str] = 'default',
                          eof: str = '__EOF__') -> _PythonTemplate:
    """
    build python template from a python function
    """
    sig = inspect.signature(py_fn)
    assert len(sig.parameters) == 1, f'{py_fn} should have only one parameter'
    args_type = sig.parameters[next(iter(sig.parameters))].annotation
    return_type  = sig.return_annotation

    fn_dir = os.path.join(base_dir, 'python/fn')
    pkg_dir = os.path.join(base_dir, 'python/pkg')
    args_file = os.path.join(fn_dir, 'args.json')
    script_path = os.path.join(fn_dir, 'script.py')
    output_parameters_dir = os.path.join(base_dir, 'output-parameters')
    input_artifacts_dir = os.path.join(base_dir, 'input-artifacts')
    output_artifacts_dir = os.path.join(base_dir, 'output-artifacts')

    dflow_input_parameters: Dict[str, dflow.InputParameter] = {}
    dflow_input_artifacts: Dict[str, dflow.InputArtifact] = {}
    dflow_output_artifacts: Dict[str, dflow.OutputArtifact] = {}
    dflow_output_parameters: Dict[str, dflow.OutputParameter] = {}

    source = [
        '#!/bin/bash',
        'set -e',
        'ARGO_PROGRESS_FILE=${ARGO_PROGRESS_FILE:-./argo_progress_file.txt}',
        '',
        setup_script,
        '',
        f'{python_cmd}  << {eof}',
        'import os, json, tarfile',
        f'base_dir = {repr(base_dir)}',
        f'fn_dir = {repr(fn_dir)}',
        f'pkg_dir = {repr(pkg_dir)}',
        f'args_file = {repr(args_file)}',
        f'script_path = {repr(script_path)}',
        f'output_parameters_dir = {repr(output_parameters_dir)}',
        f'output_artifacts_dir = {repr(output_artifacts_dir)}',
        'os.makedirs(fn_dir, exist_ok=True)',
        'os.makedirs(pkg_dir, exist_ok=True)',
        'os.makedirs(output_parameters_dir, exist_ok=True)',
        'os.makedirs(output_artifacts_dir, exist_ok=True)',
        'args = dict()',
    ]

    for f, v in iter_python_step_args(args_type):
        meta = f.type.__metadata__[0]
        if meta == types.Symbol.INPUT_PARAMETER:
            # FIXME: may have error in some corner cases
            if _is_str_type(f.type.__origin__):
                val = f'"""{{{{inputs.parameters.{f.name}}}}}"""'
            else:
                val = f'json.loads("""{{{{inputs.parameters.{f.name}}}}}""")'
            dflow_input_parameters[f.name] = dflow.InputParameter(name=f.name)
            source.append(f'args[{repr(f.name)}] = {val}')
        elif meta == types.Symbol.INPUT_ARTIFACT or isinstance(meta, dflow.InputArtifact):
            path = os.path.join(input_artifacts_dir, f.name)
            source.append(f'args[{repr(f.name)}] = {repr(path)}')
            artifact = meta
            if not isinstance(artifact, dflow.InputArtifact):
                artifact = dflow.InputArtifact(path=path, archive=default_archive)  # type: ignore
            dflow_input_artifacts[f.name] = artifact
        elif meta == types.Symbol.OUTPUT_ARTIFACT or isinstance(meta, dflow.OutputArtifact):
            path = os.path.join(output_artifacts_dir, f.name)
            source.append(f'args[{repr(f.name)}] = {repr(path)}')
            artifact = meta
            if not isinstance(artifact, dflow.OutputArtifact):
                artifact = dflow.OutputArtifact(path=path, archive=default_archive)  # type: ignore
            dflow_output_artifacts[f.name] = artifact

    source.extend([
        '',
        '# dump args to file',
        'with open(args_file, "w") as fp:',
        '    json.dump(args, fp, indent=2)',
        '',
        '# unpack tarball in pkg_dir',
        'for file in os.listdir(pkg_dir):',
        '    if file.endswith(".tar.bz2"):',
        '        with tarfile.open(os.path.join(pkg_dir, file), "r:bz2") as tar_fp:',
        '            tar_fp.extractall(pkg_dir)',
        '',
        '# insert pkg_dir to PYTHONPATH',
        'os.environ["PYTHONPATH"] = pkg_dir + ":" + os.environ.get("PYTHONPATH", "")',
        f'exit_code = os.system(f"{python_cmd} {{script_path}} {{args_file}}")',
        'assert exit_code == 0, f"python script failed with exit code {exit_code}"',
        eof,
    ])

    fn_str = [
        'import cloudpickle as cp',
        'import base64, json, bz2, os, sys',
        '',
        '# deserialize function',
        f'__fn = {pickle_converts(py_fn)}',
        '',
        '# deserialize args type',
        f'__ArgsType = {pickle_converts(args_type)}',
        '',
        '# run the function',
        'with open(sys.argv[1], "r") as fp:',
        '    __args = __ArgsType(**json.load(fp))',
        '__ret = __fn(__args)',
        '',
        '# handle the return value',
    ]

    for f, v in iter_python_step_return(return_type):
        path = os.path.join(output_parameters_dir, f.name)
        if _is_str_type(f.type.__origin__):
            fn_str.extend([
                f'with open({repr(path)}, "w") as fp:',
                f'    fp.write(str(__ret.{f.name}))'
            ])
        else:
            fn_str.extend([
                f'with open({repr(path)}, "w") as fp:',
                f'    json.dump(__ret.{f.name}, fp)'
            ])
        dflow_output_parameters[f.name] = dflow.OutputParameter(value_from_path=path)

    _dflow_script_check(fn_str, base_dir)
    return _PythonTemplate(source='\n'.join(source),
                           fn_str='\n'.join(fn_str),
                           script_path=script_path,
                           pkg_dir=pkg_dir,
                           dflow_input_parameters=dflow_input_parameters,
                           dflow_input_artifacts=dflow_input_artifacts,
                           dflow_output_parameters=dflow_output_parameters,
                           dflow_output_artifacts=dflow_output_artifacts)


def _s3_copy_fn(src, *args, **kwargs):
    if os.path.isdir(src):
        shutil.copytree(src, *args, dirs_exist_ok=True, **kwargs)
    else:
        shutil.copy2(src, *args, **kwargs)

class DFlowBuilder:
    """
    A type friendly wrapper to build a DFlow workflow.
    """

    def __init__(self, name:str, s3_prefix: str,
                 default_executor: Optional[DispatcherExecutor] = None,
                 default_setup_script: str = '',
                 default_archive: Optional[str] = 'default',
                 debug=False,
                 container_base_dir: str = '/tmp/dflow-builder',
                 allow_abs_s3_url=False,
                 s3_debug_fn = _s3_copy_fn):
        """
        :param name: The name of the workflow.
        :param s3_prefix: The base prefix of the S3 bucket to store data generated by the workflow.
        :param default_archive: The default archive method to use for Input/Output artifacts.
        :param default_executor: The default executor to run the workflow.
        :param default_setup_script: The default bash script to run at the beginning of each step.
        :param debug: If True, the workflow will be run in debug mode.
        :param local_mode: If True, the workflow will be run in local mode.
        :param container_base_dir: The base directory to mapping resources in remote container.
        :param allow_abs_s3_url: If True, allow absolute s3 url in input artifacts
        :param s3_debug_fn: The function to upload file to S3 under debug mode.
        """
        if debug:
            dflow.config['mode'] = 'debug'
            # dflow.config['debug_copy_method'] =  'copy'
            s3_prefix = s3_prefix.lstrip('/')

        assert container_base_dir.startswith('/tmp'), 'dflow: container_base_dir must start with /tmp'

        self.name: Final[str] = name
        self.workflow: Final[dflow.Workflow] = dflow.Workflow(name=name)
        self.s3_base_prefix: Final[str] = s3_prefix
        self.container_base_dir: Final[str] = container_base_dir

        self._default_archive = default_archive
        self._default_executor = default_executor
        self._default_setup_script = default_setup_script
        self._python_fns: Dict[Callable, str] = {}
        self._python_pkgs: Dict[str, str] = {}
        self._bash_scripts: Dict[Callable, str] = {}
        self._s3_cache: Dict[str, str] = {}
        self._s3_debug_fn = s3_debug_fn
        self._allow_abs_s3_url = allow_abs_s3_url
        self._debug = debug

    def s3_prefix(self, key: str):
        """
        get the full s3 prefix of a key.
        """
        base_prefix = self.s3_base_prefix
        # context prefix is inject by bohrium platform
        # to isolate namespace of different job
        context_prefix = dflow.s3_config.get('prefix')
        if context_prefix and not self._debug:
            base_prefix = os.path.join(context_prefix, base_prefix).strip('/ ')

        return os.path.join(base_prefix, key).strip('/ ')

    def s3_download(self, key: str, path: str = '.',
                    recursive: bool = True,
                    skip_exists: bool = False,
                    keep_dir: bool = False,
                    ):
        """
        Download file from S3 to local.
        """
        if self._debug:
            return
        dflow.download_s3(self.s3_prefix(key), path,  # type: ignore
                          recursive=recursive, skip_exists=skip_exists, keep_dir=keep_dir,
                          debug_func=self._s3_debug_fn)

    def s3_upload(self, path: Union[os.PathLike, str], key: str, cache: bool = False) -> str:
        """
        upload local file to S3.

        :param path: The local file path.
        :param keys: The keys of the S3 object.
        """
        if not isinstance(path, str):
            path = str(path)

        prefix = self.s3_prefix(key)
        if cache and prefix in self._s3_cache:
            return self._s3_cache[prefix]
        # FIXME: the type of dflow.upload_s3 is not correct
        self._s3_cache[prefix] = dflow.upload_s3(path, prefix, debug_func=self._s3_debug_fn)  # type: ignore
        return self._s3_cache[prefix]

    def s3_dump(self, data: Union[bytes, str], key: str) -> str:
        """
        Dump data to s3.

        :param data: The bytes to upload.
        :param keys: The keys of the S3 object.
        """
        mode = 'wb' if isinstance(data, bytes) else 'w'
        with tempfile.NamedTemporaryFile(mode) as fp:
            fp.write(data)
            fp.flush()
            return self.s3_upload(Path(fp.name), key)

    def add_step(self, step: Step):
        """
        Add a step to the workflow.
        """
        self.workflow.add(step.df_step)

    def add_steps(self, steps: Steps):
        """
        Add a list of steps to the workflow. The steps will be executed in parallel.
        """
        df_steps = _to_dflow_steps(steps)
        self.workflow.add(df_steps)

    def run(self, raise_on_failed=True):
        """
        Run the workflow.
        """
        self.workflow.submit()
        try:
            self.workflow.wait()
            if self.workflow.query_status() != 'Succeeded' and raise_on_failed:
                raise RuntimeError(f'workflow {self.name} failed')
        finally:
            if self._debug:
                resolve_ln(self.s3_base_prefix, mv=True)

    def make_bash_step(self, fn: Callable[[T_ARGS], types.ListStr], /,
                       uid: Optional[str] = None,
                       setup_script: str = '',
                       with_param: Any = None,
                       executor: Optional[DispatcherExecutor] = None,
                       ) -> Callable[[T_ARGS], Step[T_ARGS, None]]:
        """
        Make a bash step from python function.

        :param fn: The python function to generate the bash script.
        :param with_param: The parameter to pass to the step.
        :param setup_script: The bash script to run at the beginning of the step.
        :return: A function to run the step.
        """
        if uid is None:
            uid = str(uuid4())
        if not setup_script:
            setup_script = self._default_setup_script
        def wrapped_fn(args: T_ARGS):
            template = self._create_bash_template(fn, uid=uid, setup_script=setup_script)
            return self._build_step('bash-step-' + uid, args, template,
                                    with_param=with_param,
                                    executor=executor)
        return wrapped_fn

    def make_python_step(self, fn: Callable[[T_ARGS], T_RESULT], /,
                         uid: Optional[str] = None,
                         setup_script: str = '',
                         with_param: Any = None,
                         pkgs: Optional[Iterable[str]] = None,
                         executor: Optional[DispatcherExecutor] = None,
                         ) -> Callable[[T_ARGS], Step[T_ARGS, T_RESULT]]:
        """
        Make a python step.

        :param fn: The python function to run in the step.
        :param with_param: The parameter to pass to the step.
        :param setup_script: The bash script to run at the beginning of the step.
        :param pkgs: The python packages to install in the step.
        :return: A function to run the step.

        Due to the design flaw of the Argo Workflow that the s3 key cannot be set as step arguments,
        for each step a dedicated template have to be created.
        Ref: https://github.com/argoproj/argo-workflows/discussions/12606#discussioncomment-8358302
        """
        if uid is None:
            uid = str(uuid4())
        if pkgs is None:
            pkgs = ['dflow_galaxy', 'dflow', 'ai2_kit', 'jsonpickle']
        if not setup_script:
            setup_script = self._default_setup_script

        def wrapped_fn(args: T_ARGS):
            template = self._create_python_template(fn, uid=uid, pkgs=pkgs, setup_script=setup_script)
            return self._build_step('py-step-' + uid, args, template,
                                    with_param=with_param,
                                    executor=executor)
        return wrapped_fn

    def _create_bash_template(self, fn: Callable, uid: str,
                              setup_script: str = '',
                              bash_cmd: str = 'bash',):
        _template = bash_build_template(fn,
                                        base_dir=self.container_base_dir,
                                        setup_script=setup_script,
                                        default_archive=self._default_archive)
        dflow_template = ScriptOPTemplate(
            name='bash-template-' + uid,
            command=bash_cmd,
            script=_template.source,
        )
        dflow_template.inputs.parameters = _template.dflow_input_parameters
        dflow_template.inputs.artifacts = _template.dflow_input_artifacts
        dflow_template.outputs.artifacts = _template.dflow_output_artifacts
        return dflow_template

    def _create_python_template(self, fn: Callable, uid: str,
                                setup_script: str = '',
                                python_cmd: str = 'python3',
                                bash_cmd: str = 'bash',
                                pkgs: Optional[Iterable[str]] = None,
                                ):
        if pkgs is None:
            pkgs = []
        _template = python_build_template(fn, base_dir=self.container_base_dir,
                                          python_cmd=python_cmd,
                                          setup_script=setup_script,
                                          default_archive=self._default_archive)
        fn_hash = hashlib.sha256(_template.fn_str.encode()).hexdigest()
        dflow_template = ScriptOPTemplate(
            name='py-template-' + uid,
            command=bash_cmd,
            script=_template.source,
        )
        dflow_template.inputs.parameters = _template.dflow_input_parameters
        dflow_template.inputs.artifacts = _template.dflow_input_artifacts
        dflow_template.outputs.parameters = _template.dflow_output_parameters
        dflow_template.outputs.artifacts = _template.dflow_output_artifacts
        # upload python script to s3
        key = self._add_python_fn(fn, _template.fn_str, fn_hash)
        dflow_template.inputs.artifacts['__fn__'] = dflow.InputArtifact(
            source=dflow.S3Artifact(key=key),
            path=_template.script_path,
        )
        # download python packages
        for pkg in pkgs:
            key = self._add_python_pkg(pkg)
            dflow_template.inputs.artifacts[pkg] = dflow.InputArtifact(
                source=dflow.S3Artifact(key=key),
                path=os.path.join(_template.pkg_dir, f'{pkg}.tar.bz2'),
            )
        return dflow_template

    def _add_python_fn(self, fn, fn_str: str, fn_hash: str):
        if fn not in self._python_fns:
            fn_prefix = self.s3_dump(fn_str, f'build-in/python/fn/{fn_hash}')
            logger.info(f'upload {fn} to {fn_prefix}')
            self._python_fns[fn] = fn_prefix
        return self._python_fns[fn]

    def _add_python_pkg(self, pkg: str):
        """
        Add a python package to the workflow.
        """
        # Find the path of target package,
        # then create a temporary tarball and upload it to s3

        if pkg not in self._python_pkgs:
            pkg_path = os.path.dirname(__import__(pkg).__file__)
            with tempfile.NamedTemporaryFile(suffix='.tar.bz2') as fp:
                with tarfile.open(fp.name, 'w:bz2') as tar_fp:
                    tar_fp.add(pkg_path, arcname=os.path.basename(pkg_path), filter=_filter_pyc_files)
                fp.flush()
                key = self.s3_upload(Path(fp.name), f'build-in/python/pkg/{pkg}.tar.bz2')
                logger.info(f'upload python pkg {pkg} to {key}')
                self._python_pkgs[pkg] = key
        return self._python_pkgs[pkg]

    def _ensure_artifact(self, url_or_obj: DFLOW_ARTIFACT) -> DFLOW_ARTIFACT:
        if not isinstance(url_or_obj, str):
            return url_or_obj
        parsed = urlparse(url_or_obj)
        if parsed.scheme == 's3':
            key = parsed.path.lstrip('/')
            assert parsed.netloc in ('', '.'), f'unsupported s3 url {url_or_obj}'
            if '.' == parsed.netloc:
                key = self.s3_prefix(key)
            elif not self._allow_abs_s3_url:
                raise ValueError(f'absolute s3 url {url_or_obj} is not allowed, use relative path instead, or set allow_abs_s3_url=True')
            if self._debug:
                key = os.path.abspath(key)
            return dflow.S3Artifact(key=key)
        raise ValueError(f'unsupported url {url_or_obj}')

    def _build_step(self, name: str, args: T_ARGS, template,
                    with_param: Any=None,
                    executor: Optional[DispatcherExecutor] = None):
        if executor is None:
            executor = self._default_executor

        parameters = {}
        artifacts = {}
        for f, v in iter_python_step_args(args):
            meta = f.type.__metadata__[0]
            if meta == types.Symbol.INPUT_PARAMETER:
                parameters[f.name] = v
            elif meta == types.Symbol.INPUT_ARTIFACT or isinstance(meta, dflow.InputArtifact):
                artifacts[f.name] = self._ensure_artifact(v)  # type: ignore
            elif meta == types.Symbol.OUTPUT_ARTIFACT or isinstance(meta, dflow.OutputArtifact):
                template.outputs.artifacts[f.name].save = [self._ensure_artifact(v)]  # type: ignore
            else:
                raise ValueError(f'unsupported type {f.type}')

        step = dflow.Step(
            name=name,
            template=template,
            with_param=with_param,
            parameters=parameters,
            artifacts=artifacts,
            executor=executor,
        )
        return Step(step)


def _filter_pyc_files(tarinfo):
    if tarinfo.name.endswith('.pyc') or tarinfo.name.endswith('__pycache__'):
        return None
    return tarinfo


def _to_dflow_steps(steps: Steps):
    if isinstance(steps, Step):
        return steps.df_step
    return [_to_dflow_steps(step) for step in steps]


def _dflow_script_check(source: Iterable[str], base_dir):
    """
    ensure the script is safe to run in dflow
    """
    for line in source:
        assert '/tmp' not in line.replace(base_dir, ''), 'dflow: script should not contain /tmp literal'
        assert '"""' not in line, 'dflow: script should not contain """'
        assert '$(pwd)' not in line, 'dflow: script should not contain $(pwd)'

def _is_str_type(cls):
    # FIXME: this may have problem in some corner case
    try:
        return issubclass(cls, str)
    except TypeError:
        return False