from typing import Optional, TypeVar
from ai2_kit.core.util import list_split
import sys
import os


from .types import ListStr, SliceIndex

T = TypeVar('T')

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_dirname(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def not_none(v: Optional[T], msg: str = '') -> T:
    if v is None:
        raise ValueError(msg)
    return v


def ensure_str(s: ListStr):
    if isinstance(s, list):
        return '\n'.join(s)
    return s


def bash_ln_cmd(from_path: str, to_path: str):
    """
    The reason to `rm -d` to_path is to workaround the limit of ln.
    `ln` command cannot override existed directory,
    so we need to ensure to_path is not existed.
    Here we use -d option instead of -rf to avoid remove directory with content.
    The error of `rm -d` is suppressed as it will fail when to_path is file.
    `-T` option of `ln` is used to avoid some unexpected result.
    """
    return f'rm -d {to_path} || true && ln -sfT {from_path} {to_path}'


def safe_ln(from_path: str, to_path: str, method=None):
    if method is None:
        method = os.system
    method(bash_ln_cmd(from_path, to_path))


def select_chunk(in_list: list, n: int, i: int):
    assert 0 <= i < n, f'nth should be in range [0, {n})'
    return list_split(sorted(in_list), n)[i]


def inspect_dir(dir: str):
    os.system(bash_inspect_dir(dir))


def bash_inspect_dir(dir: str):
    return f"""echo "Show content of {dir}" && find {dir}"""


def bash_iter_ls_slice(search_pattern: str, /, n: int, i: SliceIndex, script: ListStr, opt: str = '',
                       it_var='ITEM', python_cmd: str = 'python'):
    """
    Generate a bash snippet to slice the result of `ls` command,
    and iterate over the selected chunk.

    :param search_pattern: search pattern for directories
    :param n: number of chunks
    :param i: chunk index
    :param script: bash script to process each directory
    :param it_var: variable name for each item
    """
    return '\n'.join([
        f'_LS_RESULT=$(ls -1 {opt} {search_pattern} | sort)',
        bash_slice(in_var='_LS_RESULT', n=n, i=i, out_var='_LS_CHUNK', python_cmd=python_cmd),
        bash_iter_var(in_var='_LS_CHUNK', script=script, it_var=it_var),
    ])


def bash_iter_var(in_var: str, script: ListStr, it_var='ITEM', report_progress=True):
    """
    Generate a bash snippet to iterate over lines of a variable

    :param in_var: variable name of input data
    :param script: bash script to process each line
    """
    script = ensure_str(script)

    argo_progress_init, argo_progress_count  = '', ''
    if report_progress:
        argo_progress_init = f'_ARGO_I=0 && echo "$_ARGO_I/$_N_LINES" > $ARGO_PROGRESS_FILE'
        argo_progress_count = '_ARGO_I=$((_ARGO_I + 1)) && echo "$_ARGO_I/$_N_LINES" > $ARGO_PROGRESS_FILE'

    return f"""_N_LINES=$(grep . <<< "${in_var}" | wc -l)
{argo_progress_init}
if [ $_N_LINES -ne 0 ]; then
while IFS= read -r {it_var}; do
{script}
{argo_progress_count}
done <<< "${in_var}"
fi"""



def bash_slice(in_var: str, n: int, i: SliceIndex, out_var: str,
               python_cmd: str = 'python'):
    """
    Generate a bash snippet to slice a multi-line string variable
    into n chunks and select the ith chunk

    :param in_var: variable name of input multi-line string
    :param n: number of chunks
    :param i: chunk index
    :param out_var: variable name to store the selected chunk
    """
    return f"""# bash_slice({in_var}, {n}, {i}, {out_var})
which python3 && PY_CMD=python3 || PY_CMD=python  # prefer python3
which "{python_cmd}" && PY_CMD="{python_cmd}" || true  # prefer user specified python
{out_var}=$(_IN_DATA="${in_var}" _SLICE_N={n} _SLICE_I={i} $PY_CMD << EOF
import sys,os
lines = os.environ['_IN_DATA'].split('\\n')
n = int(os.environ['_SLICE_N'])
i = int(os.environ['_SLICE_I'])
lines = [line for line in lines if line.strip()]
chunk_size = max(1, len(lines) // n)

start = i * chunk_size
end = (i + 1) * chunk_size if i < n - 1 else len(lines)
sys.stdout.write('\\n'.join(lines[start:end]))
EOF
)
# bash_slice end"""


def yes_or_no(msg: str, default: bool = False):
    """
    prompt user for input, if user press n or N, return False
    if user press y or Y, return True
    otherwise, ask again
    """
    msg = f'{msg} [Y/n]: ' if default else f'{msg} [n/Y]: '

    while True:
        ans = input(msg)
        if not ans:
            return default
        if ans.lower() == 'y':
            return True
        if ans.lower() == 'n':
            return False
        print('Invalid input, please input y or n')


def resolve_ln(path: str, mv=False):
    """
    This command will iterate over the files in the given directory recursively,
    if the file is a soft link, it will resolve the link and
    copy the resolved file to the same directory with the same name.

    :param path: the directory to resolve symlinks
    :param mv: if True, move the resolved file to the same directory with the same name
    """
    if os.path.islink(path):
        resolved_path = os.path.realpath(path)
        resolved_file = os.path.join(os.path.dirname(path), os.path.basename(resolved_path))
        cmd = 'mv' if mv else 'cp -f'
        os.system(f'rm {resolved_file} && {cmd} {resolved_path} {resolved_file}')
        # log is too verbose
        # print(f'replace {path} with {resolved_path}', file=sys.stderr)

    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                resolve_ln(file_path, mv=mv)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                resolve_ln(dir_path, mv=mv)


def parse_string_array(s: str, dtype=None, delimiter=None):
    arr = [x.strip() for x in s.split(delimiter)]
    if dtype:
        arr = [dtype(x) for x in arr]
    return arr

def str_or_none(s):
    """
    return None if s is empty string
    """
    if not s:
        return None
    return s
