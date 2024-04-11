from dp.launching.typing import BaseModel, Field, OutputDirectory, InputFilePath
from dp.launching.typing import Int, String, Enum, Float, Boolean, List, Optional, Dict
from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.report import Report, ReportSection, ChartReportElement

from dflow_galaxy.app.common import DFlowOptions, setup_dflow_context, EnsembleOptions
from dflow_galaxy.res import get_res_path
from dflow_galaxy.core.log import get_logger

from dflow_galaxy.workflow.tesla.main import build_tesla_workflow

from ai2_kit.core.util import dump_json, load_text, load_json
from ai2_kit.feat import catalysis as ai2cat

import ase.io
import fire

from pathlib import Path
from uuid import uuid4
import shutil
import glob
import sys
import os


logger = get_logger(__name__)


BH_DEEPMD_DEFAULT = {
    'image_name': 'registry.dp.tech/dptech/dpmd:2.2.8-cuda12.0',
    'scass_type': 'c8_m32_1 * NVIDIA V100',
}
BH_LAMMPS_DEFAULT = BH_DEEPMD_DEFAULT

BH_CP2K_DEFAULT = {
    'image_name': 'registry.dp.tech/dptech/cp2k:11',
    'scass_type': 'c32_m64_cpu',
}

BH_PYTHON_DEFAULT = {
    'image_name': 'registry.dp.tech/dptech/prod-13325/dflow-galaxy:0.1.4-main-8bb98c7',
    'scass_type': 'c2_m4_cpu',
}


class KvItem(BaseModel):
    key: String = Field(
        description="Key of the item")
    value: String = Field(
        format='multi-line',
        description="Value of the item")


class ExploreItem(BaseModel):
    key: String = Field(
        description="Key of the item")
    value: String = Field(
        description="Value of the item, multiple value should be separated by comma")
    broadcast: Boolean = Field(
        default=False,
        description="Use broadcast instead of full combination")


class DeepmdSettings(BaseModel):

    compress_model: Boolean = Field(
        default=True,
        description="Compress the model after training")

    concurrency: Int = Field(
        default=4,
        description="Number of concurrent run")

    cmd: String = Field(
        default='dp',
        description="Command to run DeepMD, note that it depends on the docker image you used")

    setup_script: String = Field(
        default='',
        format='multi-line',
        description="Setup script for DeepMD simulation, note that it depends on the docker image you used")


class LammpsSetting(BaseModel):
    ensemble: EnsembleOptions = Field(
        default=EnsembleOptions.csvr,
        description='Ensemble of LAMMPS simulation')

    plumed_config: Optional[String] = Field(
        format='multi-line',
        description='Plumed configuration file for metadynamics simulation')

    explore_vars: List[ExploreItem] = Field(
        default=[
           ExploreItem(key='TEMP', value='50,100,200,300,400,600,800,1000', broadcast=False),
           ExploreItem(key='PRES', value='1', broadcast=False),
        ],
        description="Variables for LAMMPS exploration, TEMP, PRES are required, you may also add TAU_T, TAU_P, etc")

    template_vars: List[KvItem] = Field(
        default=[
            KvItem(key='POST_INIT', value='\n'.join([
                'neighbor 1.0 bin',
                'box      tilt large',
            ])),
            KvItem(key='POST_READ_DATA', value='\n'.join([
                'change_box all triclinic'
            ])),
        ],
        description="Template variables for LAMMPS exploration, you may need to modify this according to your system")

    nsteps: Int = Field(
        default=1000,
        description='Number of steps of LAMMPS simulation for the first iteration')

    timestep: Float = Field(
        default=0.0005,
        description='Time step size of LAMMPS simulation in ps')

    sample_freq: Int = Field(
        default=10,
        description='Sampling frequency of LAMMPS simulation')

    no_pbc: Boolean = Field(
        default=False,
        description='Whether to use periodic boundary condition')

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    cmd: String = Field(
        default='lmp',
        description="Command to run LAMMPS, note that it depends on the docker image you used")

    setup_script: String = Field(
        default='',
        format='multi-line',
        description="Setup script for LAMMPS simulation, note that it depends on the docker image you used")


class ModelDeviation(BaseModel):
    lo: Float = Field(
        default=0.2,
        description="Lower bound of the deviation")

    hi: Float = Field(
        default=0.6,
        description="Upper bound of the deviation")

    metric: String = Field(
        default='max_devi_f',
        description="Metric to measure the deviation")


class Cp2kSettings(BaseModel):

    limit: Int = Field(
        default=50,
        description="Limit of the number of structures to be labeled for each iteration")

    concurrency: Int = Field(
        default=5,
        description="Number of concurrent run")

    cmd: String = Field(
        default='mpirun -np 32 cp2k.popt',
        description="Script to run CP2K simulation, note that it depends on the docker image")

    setup_script: String = Field(
        default = '\n'.join([
            '# guess cp2k data dir',
            '[[ -z "${CP2K_DATA_DIR}" ]] && export CP2K_DATA_DIR="$(dirname "$(which cp2k || which cp2k.psmp)")/../../data" || true',
            'source /opt/cp2k-toolchain/install/setup',
        ]),
        format='multi-line',
        description="Setup script for CP2K simulation, note that it depends on the docker image you used")


class DynacatTeslaArgs(DFlowOptions):
    deepmd_dataset : InputFilePath = Field(
        title='DeepMD Dataset',
        description="DeepMD in zip or tgz format")

    deepmd_input_template: InputFilePath = Field(
        title='DeepMD Input Template',
        description="Input template file for DeepMD training")

    lammps_system_file: InputFilePath = Field(
        description="Structure file in xyz format use for LAMMPS simulation")

    cp2k_input_template: InputFilePath = Field(
        description="Input template file for CP2K simulation")

    dry_run: Boolean = Field(
        default = True,
        description="Generate configuration file without running the simulation")

    max_iters: Int = Field(
        default = 7,
        description="Maximum iterations of the workflow")

    s3_prefix: Optional[String] = Field(
        description="Specify the S3 prefix of DFlow. By default a random prefix will be used for different jobs. Jobs use the same prefix will share the same working directory, which allow you to inherit the state of previous run.")

    deepmd: DeepmdSettings = Field(
        description="DeepMD settings for training")

    lammps: LammpsSetting = Field(
        description="LAMMPS settings for structure exploration")

    model_deviation: ModelDeviation = Field(
        description="Model deviation settings for screening structures")

    cp2k: Cp2kSettings = Field(
        description="CP2K settings for DFT labeling")

    output_dir : OutputDirectory = Field(
        default='./output',
        description="Output directory of LAMMPS simulation")

    deepmd_resource: Dict[String, String] = Field(
        default=BH_DEEPMD_DEFAULT,
        description="Resource configuration for DeepMD")

    lammps_resource: Dict[String, String] = Field(
        default=BH_LAMMPS_DEFAULT,
        description="Resource configuration for LAMMPS")

    cp2k_resource: Dict[String, String] = Field(
        default=BH_CP2K_DEFAULT,
        description="Resource configuration for CP2K")

    python_resource: Dict[String, String] = Field(
        default=BH_PYTHON_DEFAULT,
        description="Resource configuration for Python")


def launch_app(args: DynacatTeslaArgs) -> int:
    s3_prefix = args.s3_prefix or f'dynacat-{uuid4()}'
    logger.info(f'using s3 prefix: {s3_prefix}')
    tesla_template = get_res_path() / 'dynacat' / 'tesla_template.yml'
    # handle deepmd dataset
    dp_dataset_file = args.deepmd_dataset.get_full_path()
    dp_dataset = _unpack_dpdata(dp_dataset_file, 'init-dataset')
    dp_dataset_config = {}
    for i, d in enumerate(dp_dataset):
        dp_dataset_config[f'dpdata-{i}'] = {
            'url': d,
        }
    logger.info(f'Unpacked dpdata to {dp_dataset}')

    # build config
    shutil.copy(tesla_template, 'tesla-preset.yml')
    executor_config = _get_executor_config(args)
    workflow_config = _get_workflow_config(args, dp_dataset_config)

    # TODO: use yaml to dump config pretty
    dump_json(executor_config, 'tesla-executor.yml')
    dump_json(workflow_config, 'tesla-workflow.yml')

    # copy generated configuration to output
    os.makedirs(args.output_dir, exist_ok=True)
    os.system(f'cp tesla-*.yml {args.output_dir}')

    if args.dry_run:
        logger.info('Skip running workflow due to dry_run is set to True')
        return 0

    setup_dflow_context(args)
    workflow = build_tesla_workflow('tesla-preset.yml', 'tesla-executor.yml', 'tesla-workflow.yml',
                                    s3_prefix=s3_prefix, max_iters=args.max_iters)
    try:
        workflow.run()
    except:
        logger.exception('Failed to run workflow')
        return 1
    finally:
        # reclaim useful data
        dp_dataset_dir = os.path.join(args.output_dir, 'dp-dataset')
        dp_models_dir = os.path.join(args.output_dir, 'dp-models')
        model_devi_dir = os.path.join(args.output_dir, 'model-devi')
        os.makedirs(dp_dataset_dir, exist_ok=True)
        os.makedirs(dp_models_dir, exist_ok=True)
        os.makedirs(model_devi_dir, exist_ok=True)
        workflow.s3_download('iter-dataset', dp_dataset_dir)
        workflow.s3_download('train-deepmd', dp_models_dir)
        workflow.s3_download('screen-model-devi', model_devi_dir)
        _gen_report(dp_models_dir=dp_models_dir,
                    model_devi_dir=model_devi_dir,
                    max_iters=args.max_iters,
                    output_dir=str(args.output_dir))
    return 0


def _gen_report(dp_models_dir: str,
                model_devi_dir: str,
                max_iters: int,
                output_dir: str):
    # build report sections, each iter per section, from last to first
    sections = []
    for i in reversed(range(max_iters)):
        iter_str = f'iter/{i:03d}'
        lcurve_files = glob.glob(f'{dp_models_dir}/{iter_str}/**/lcurve.out', recursive=True)
        model_devi_files = glob.glob(f'{model_devi_dir}/{iter_str}/**/report.tsv', recursive=True)

        if not (lcurve_files or model_devi_files):
            logger.info(f'No data found for iteration {i}')
            continue
        sections.append(_gen_report_section(i, lcurve_files, model_devi_files))
    # write report
    report = Report(title='DynaCat TESLA', sections=sections)
    report.save(output_dir)


def _gen_report_section(iter: int, lcurve_files: List[str], model_devi_files: List[str]):
    elements = []
    if lcurve_files:
        for i, f in enumerate(sorted(lcurve_files)):
            name = os.path.dirname(f)
            echart = _gen_lcurve_echart(f)
            element = ChartReportElement(
                title=f'Learning Curve of training: {name}',
                options=echart,
            )
            elements.append(element)

    if model_devi_files:
        # there should be only 1 file in each iteration
        f = model_devi_files[0]
        echart = _gen_model_devi_stats_echart(f)
        element = ChartReportElement(
            title='Model Deviation Statistics',
            options=echart,
        )
        elements.append(element)

    section = ReportSection(
        title=f'Result of Iteration {iter:03d}',
        ncols=2,
        elements=elements,
    )
    return section


def _gen_model_devi_stats_echart(file: str):
    data_dict = _load_model_devi_stats(file)
    echart = {
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            }
        },
        'legend': {
            'data': ['Good', 'Decent', 'Poor']
        },
        'grid': {
            'left': '10%',
            'right': '10%',
            'bottom': '3%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'value'
        },
        'yAxis': {
            'type': 'category',
            'data': data_dict['src']
        },
        'series': [
            {
                'name': 'Good',
                'type': 'bar',
                'stack': 'total',
                'label': {
                    'show': False,
                },
                'itemStyle': {
                    'color': '#67C23A'  # Green color to indicate 'good' is better
                },
                'data': [int(d) for d in data_dict['good']]
            },
            {
                'name': 'Decent',
                'type': 'bar',
                'stack': 'total',
                'label': {
                    'show': False
                },
                'itemStyle': {
                    'color': '#E6A23C'  # Orange color to indicate 'decent'
                },
                'data': [int(d) for d in data_dict['decent']]
            },
            {
                'name': 'Poor',
                'type': 'bar',
                'stack': 'total',
                'label': {
                    'show': False
                },
                'itemStyle': {
                    'color': '#F56C6C'  # Red color to indicate 'poor' (danger)
                },
                'data': [int(d) for d in data_dict['poor']]
            }
        ]
    }
    return echart


def _load_model_devi_stats(file: str):
    header = None
    with open(file, newline='') as fp:
        headers = _parse_string_array(next(fp), delimiter='\t')
        data_dict = {name: [] for name in headers}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            values = _parse_string_array(line, delimiter='\t')
            for i, header in enumerate(headers):
                data_dict[header].append(values[i])
    return data_dict


def _gen_lcurve_echart(file: str):
    series = _load_lcurve(file)
    x = series.pop('step')
    echart = {
        'tooltip': {
            'trigger': 'axis',
        },
        'xAxis': {
            'type': 'category',
            'name': 'Step',
            'data': x,
        },
        'yAxis': [
            {
                'type': 'log',
                'name': 'RMSE',
                'position': 'left',
            },
            {
                'type': 'log',
                'name': 'Learning Rate',
                'position': 'right',
            }
        ],
        'legend': {
            'data': [name for name in series.keys()],
        },
        'series': [],
    }
    for name, data in series.items():
        echart['series'].append({
            'name': name,
            'data': data,
            'type': 'line',
            'smooth': True,
            'yAxisIndex': 0 if name != 'lr' else 1,
        })
    return echart


def _load_lcurve(file: str):
    header = None
    data = []
    with open(file, encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if header is not None:
                    continue  # ignore comment line
                header = line[1:].split()
            else:
                data.append([float(x) for x in line.split()])
    assert header is not None, 'Failed to parse lcurve file'
    # convert to series
    series = {}
    for i, h in enumerate(header):
        series[h] = [d[i] for d in data]
    return series


def _get_executor_config(args: DynacatTeslaArgs):
    return {
        'executors': {
            'bohrium': {
                'bohrium': {},
                'apps': {
                    'python': {
                        'resource': {
                            'bohrium': {**args.python_resource, 'job_name':'dynacat_tesla_python'},
                        }
                    },
                    'deepmd': {
                        'resource': {
                            'bohrium': {**args.deepmd_resource, 'job_name':'dynacat_tesla_deepmd'},
                        },
                        'dp_cmd': args.deepmd.cmd,
                        'concurrency': args.deepmd.concurrency,
                        'setup_script': args.deepmd.setup_script,
                    },
                    'lammps': {
                        'resource': {
                            'bohrium': {**args.lammps_resource, 'job_name':'dynacat_tesla_lammps'},
                        },
                        'lammps_cmd': args.lammps.cmd,
                        'concurrency': args.lammps.concurrency,
                        'setup_script': args.lammps.setup_script,
                    },
                    'cp2k': {
                        'resource': {
                            'bohrium': {**args.cp2k_resource, 'job_name': 'dynacat_tesla_cp2k'},
                        },
                        'cp2k_cmd': args.cp2k.cmd,
                        'concurrency': args.cp2k.concurrency,
                        'setup_script':  args.cp2k.setup_script,
                    }
                }
            }
        },
        'orchestration': {
            'deepmd': 'bohrium',
            'lammps': 'bohrium',
            'model_devi': 'bohrium',
            'cp2k': 'bohrium'
        },
    }


def _get_workflow_config(args: DynacatTeslaArgs, dp_dataset_config: dict):
    # process system file
    explore_data_file = args.lammps_system_file.get_full_path()
    explore_data_key = os.path.basename(explore_data_file)
    atoms = ase.io.read(explore_data_file, index=0)
    type_map, mass_map = ai2cat.get_type_map(atoms)  # type: ignore

    cp2k_input_template = load_text(args.cp2k_input_template.get_full_path())
    deepmd_template = load_json(args.deepmd_input_template.get_full_path())
    product_vars, broadcast_vars = _get_lammps_vars(args.lammps.explore_vars)

    return {
        'datasets': {
            **dp_dataset_config,
            explore_data_key: {  # data key must be a normal file name or else ase cannot detect the format
                'url': explore_data_file,
            },
        },
        'workflow': {
            'general': {
                'type_map': type_map,
                'mass_map': mass_map,
            },
            'train': {
                'deepmd': {
                    'init_dataset': list(dp_dataset_config.keys()),
                    'input_template': deepmd_template,
                    'compress_model': args.deepmd.compress_model,
                }
            },
            'explore':{
                'lammps': {
                    'systems': [explore_data_key],
                    'nsteps': args.lammps.nsteps,
                    'ensemble': args.lammps.ensemble.value,
                    'timestep': args.lammps.timestep,
                    'sample_freq': args.lammps.sample_freq,
                    'no_pbc': args.lammps.no_pbc,
                    'plumed_config': args.lammps.plumed_config or None,
                    'product_vars': product_vars,
                    'broadcast_vars': broadcast_vars,
                    'template_vars': dict((item.key, item.value) for item in args.lammps.template_vars),
                }
            },
            'screen':{
                'model_devi': {
                    'metric': args.model_deviation.metric,
                    'decent_range': [args.model_deviation.lo, args.model_deviation.hi],
                }
            },
            'label': {
                'cp2k': {
                    'input_template': cp2k_input_template,
                    'limit': args.cp2k.limit,
                }
            },
        }
    }


def _get_lammps_vars(explore_vars: List[ExploreItem]):
    broadcast_vars = {}
    product_vars = {}
    for item in explore_vars:
        if item.broadcast:
            broadcast_vars[item.key] = _parse_string_array(item.value, dtype=float)
        else:
            product_vars[item.key] = _parse_string_array(item.value, dtype=float)
    return product_vars, broadcast_vars


def _parse_string_array(s: str, dtype=None, delimiter=','):
    arr = [x.strip() for x in s.split(delimiter)]
    if dtype:
        arr = [dtype(x) for x in arr]
    return arr


def _unpack_dpdata(file: str, extract_dir: str):
    extract_dir = os.path.normpath(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    shutil.unpack_archive(file, extract_dir=extract_dir)
    # use type.raw to locate the dpdata folder
    paths = glob.glob(f'{extract_dir}/**/type.raw', recursive=True)
    return [ os.path.dirname(p) for p in paths]


def main():
    to_runner(
        DynacatTeslaArgs,
        launch_app,
        version="0.1.0",
    )(sys.argv[1:])


if __name__ == "__main__":
    fire.Fire({
        'main': main,
        'generate_report': _gen_report,
    })
