import argparse
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def one_hot_int(num_classes: int = 10,
                dtype: torch.dtype = torch.float32) -> Callable[[int], torch.Tensor]:
    """
    Return a function that performs one hot encoding.

    :param num_classes:
        number of classes for one hot encoder.
    :param dtype:
        data type for encoder output.

    :return:
        one hot encoder for integers with params.
    """

    def function(value: int) -> torch.Tensor:
        """
        :return:
            one hot encoding of value as pytorch tensor.
        """

        # int to tensor
        tensor_value = torch.tensor(value)

        # one-hot encoding
        one_hot_label = F.one_hot(input=tensor_value,
                                  num_classes=num_classes)
        # assign specific data type
        return one_hot_label.to(dtype)

    return function


def cosine_decay(steps: int) -> Callable[[int], float]:
    """
    Returns a function that performs cosine decay.

    :param steps:
        maximum number of steps considered for decay.
    :return:
        cosine decay function.
    """

    def function(step: int) -> float:
        """
        :return:
            factor that will decay another value.
        """

        arg = 7 * np.pi * step / (16 * steps)
        return np.cos(arg)

    return function


def options(fixmatch: bool = False):
    """
    Controls and parse CLI arguments from user.

    :param fixmatch:
        activates additional options just for fixmatch.

    :return:
        NameSpace with user input args.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="dataset for training",
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')
    parser.add_argument("-e", "--epochs",
                        help="number of epochs for training",
                        type=int,
                        default=30)
    parser.add_argument("-bs", "--batch-size",
                        help="batch size for training",
                        type=int,
                        default=128)
    parser.add_argument("-md", "--model-depth",
                        help="depth of Wide ResNet model",
                        type=int,
                        default=28)
    parser.add_argument("-mw", "--model-width",
                        help="width of Wide ResNet model",
                        type=int,
                        default=2)
    parser.add_argument("-es", "--early-stopping",
                        help="number of epochs for early stopping",
                        type=int,
                        default=15)
    parser.add_argument("-r", "--results",
                        help="folder name for training results",
                        default="alldata")
    parser.add_argument("-tr", "--train-range",
                        help="range of images per class for training",
                        nargs=2,
                        type=int,
                        default=[0, 4000])
    parser.add_argument("-vr", "--val-range",
                        help="range of images per class for validation",
                        nargs=2,
                        type=int,
                        default=[4000, 5000])

    if fixmatch:
        parser.add_argument("-ulr", "--unlabeled-range",
                            help="range of images per class for unlabeled data",
                            nargs=2,
                            type=int,
                            default=[0, 4000])
        parser.add_argument("-tau", "--tau",
                            help="threshold for retaining a pseudo-label",
                            type=float,
                            default=0.95)
        parser.add_argument("-mu", "--mu",
                            help="multiplier of batch size for unlabeled data",
                            type=float,
                            default=7.)
        parser.add_argument("--lambda-u",
                            help="unsupervised loss multiplier lambda",
                            type=float,
                            default=1.)
        parser.add_argument("-N", "--N",
                            help="number of transformations for RandAugment",
                            type=int,
                            default=2)
        parser.add_argument("-M", "--M",
                            help="magnitude of transformations in RandAugment",
                            type=int,
                            default=9)

    return parser.parse_args()
