from typing import Tuple

import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10SSL(CIFAR10):
    """
    Dataset CIFAR10 adapted for semi-supervised learning.

    Adapts CIFAR10 dataset provided by PyTorch by allowing a
    range which selects a subset of images for each class.
    """

    def __init__(self,
                 root_path: str,
                 train: bool,
                 data_range: Tuple[int, int]):
        """
        Initialize a CIFAR10SSL datset object.

        :param root_path:   path to store or look up data
        :param train:       get data from train set (50000 images) or from test set (10000 images)
        :param data_range:  range of images to select from every class
        """

        super(CIFAR10SSL, self).__init__(root=root_path,
                                         train=train,
                                         download=True)

        self.__order_data()
        self.data_range = data_range
        self.__resolve_data_range()
        self.shuffle_data()

    def shuffle_data(self):
        """
        Shuffle both data and its targets randomly.
        """

        permutation = np.random.permutation(len(self))

        self.targets = self.targets[permutation]
        self.data = self.data[permutation]

    def __order_data(self):
        """
        Order data sequentially by class index (0 to 9).
        """

        # get order from targets then apply it to data and targets
        order = np.asarray(self.targets).argsort()
        self.targets = np.asarray(self.targets)[order]
        self.data = self.data[order]

    def __resolve_data_range(self):
        """
        Select a reduced number of examples per class using a range.
        """

        # check if data_range is a tuple with 2 elements
        assert isinstance(self.data_range, tuple), 'range must be a tuple'
        assert len(self.data_range) == 2, 'range tuple must have exactly two integers'

        # check if both data_range elements are integers
        range_l, range_h = self.data_range
        assert isinstance(range_l, int) and isinstance(range_h, int), 'numbers must be integers'

        # check if data_range is within boundaries
        max_val = 5000 if self.train else 1000
        assert 0 <= range_l < range_h <= max_val, f'invalid range (max {max_val} img/class)'

        # select data and targets specified by validated data_range
        ids = [idx + max_val * nclass for nclass in range(len(self.classes)) for idx in range(range_l, range_h)]
        self.data = self.data[ids]
        self.targets = self.targets[ids]


if __name__ == "__main__":
    cifar10ds = CIFAR10SSL(root_path='./cifar10', train=True, data_range=(3, 5))
    print('Cantidad de datos en dataset: ', len(cifar10ds))
