from pathlib import Path

def get_cp2k_data_dir():
    return Path(__file__).parent / "cp2k_data"
