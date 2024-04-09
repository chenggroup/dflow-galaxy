from pathlib import Path

def get_res_path():
    return Path(__file__).parent

def get_cp2k_data_dir():
    return get_res_path() / "cp2k_data"

