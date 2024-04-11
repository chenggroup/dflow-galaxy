from dp.launching.report import Report, ReportSection, ChartReportElement
from typing import List
import glob
import os

from dflow_galaxy.core.util import parse_string_array


def gen_report(lammps_output_dir: str,
               output_dir: str):
    sections = []
    fes_file = os.path.join(lammps_output_dir, 'fes.dat')



    # write report
    report = Report(title='DynaCat MD', sections=sections)
    report.save(output_dir)

def _gen_fes_echart(f: str):
    data_points = _load_fes_data(f)
    echart = {
        'title': {
            'text': 'Data Visualization'
        },
        'tooltip': {
            'trigger': 'axis'
        },
        'xAxis': {
            'type': 'value',
            'name': 'x-axis'
        },
        'yAxis': {
            'type': 'value',
            'name': 'y-axis'
        },
        'series': [{
            'data': data_points,
            'type': 'line'
        }]
    }
    return echart



def _load_fes_data(f: str):
    """
    #! FIELDS d1 file.free der_d1
    #! SET min_d1 0.273758
    #! SET max_d1 6.8191
    #! SET nbins_d1  94
    #! SET periodic_d1 false
    0.273758000    0.998068539   -0.000000000
    0.344138022    0.997859772   -0.008132961
    0.414518043    0.996330768   -0.040475018
    0.484898065    0.991143958   -0.119637947
    0.555278086    0.977356216   -0.288794464

    get fields from first row, ignore other comments, and parse data
    """
    field_prefix = '#! FIELDS'
    header = None
    rows = []
    with open(f, encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if line.startswith(field_prefix):
                header = parse_string_array(line[len(field_prefix):], delimiter=' ')
            elif line.startswith('#'):
                continue
            else:
                rows.append(parse_string_array(line, dtype=float, delimiter=' '))
    # convert to data_dict
    data_dict = {}
    assert header is not None, "Invalid fes file: missing header line"
    for i, header in enumerate(header):
        data_dict[header] = [row[i] for row in rows]





    ...