"""
definitions.py:
    This file is at the root of the project and contains essential data that remains constant.
"""

import os
from pathlib import Path

"""
Relevant paths for the project.
    ROOT_DIR - root path of the project
    SOURCES_DIR - path where all source code is located
    DATASETS_DIR - path where to save dataset CIFAR
    RUNS_DIR - path where to save results from model trainings
"""
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SOURCES_DIR = Path(ROOT_DIR) / 'src'
DATASETS_DIR = Path(ROOT_DIR) / 'datasets'
RUNS_DIR = Path(ROOT_DIR) / 'runs'

"""
Dataset CIFAR both 10 and 100 mean and standard deviation.

Note that these where pre-computed over all 50000 images of training data
but using the float representation of the images which maps every pixel
value from 0 to 255 to a float value in the range 0 to 1.
"""
CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)

CIFAR100_MEAN = (0.50707516, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733429, 0.25643846, 0.27615047)
