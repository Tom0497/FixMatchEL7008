import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SOURCES_DIR = Path(ROOT_DIR) / 'src'
DATASETS_DIR = Path(ROOT_DIR) / 'datasets'
RUNS_DIR = Path(ROOT_DIR) / 'runs'
