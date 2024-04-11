from dp.launching.report import Report, ReportSection, ChartReportElement
from typing import List
import glob
import os

from dflow_galaxy.core.util import parse_string_array

def gen_report(dp_models_dir: str,
               model_devi_dir: str,
               max_iters: int,
               output_dir: str):
    # build report sections, each iter per section, from last to first
    os.makedirs(output_dir, exist_ok=True)
    sections = []
    for i in reversed(range(max_iters)):
        iter_str = f'iter/{i:03d}'
        lcurve_files = glob.glob(f'{dp_models_dir}/{iter_str}/**/lcurve.out', recursive=True)
        model_devi_files = glob.glob(f'{model_devi_dir}/{iter_str}/**/report.tsv', recursive=True)

        if not (lcurve_files or model_devi_files):
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
        headers = parse_string_array(next(fp), delimiter='\t')
        data_dict = {name: [] for name in headers}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            values = parse_string_array(line, delimiter='\t')
            for i, header in enumerate(headers):
                data_dict[header].append(values[i])
    return data_dict


def _gen_lcurve_echart(file: str):
    data_dict = _load_lcurve(file)
    x = data_dict.pop('step')
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
            'data': [name for name in data_dict.keys()],
        },
        'series': [],
    }
    for name, data in data_dict.items():
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