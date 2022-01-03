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
