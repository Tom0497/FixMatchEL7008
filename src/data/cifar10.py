from typing import Tuple, Optional, Callable

import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10SSL(CIFAR10):
    """
    Dataset CIFAR10 adapted for semi-supervised learning.

    Adapts CIFAR10 dataset class provided by PyTorch by allowing a
    range which selects a subset of images for each class.

    CIFAR-10 dataset consists of 60k 32x32 colour images, from 10
    classes, 6k images per class. 50k are training images and 10k
    are test images. The home page of the dataset is provided
    in <https://www.cs.toronto.edu/~kriz/cifar.html> by the
    University of Toronto.
    """

    def __init__(self,
                 root_path: str,
                 train: bool,
                 data_range: Tuple[int, int],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Constructor of CIFAR10SSL dataset object.

        :param root_path:
            path to store or look up data.
        :param train:
            get data from train set (50000 images) or from test set (10000 images).
        :param data_range:
            range of images to select from every class.
        """

        super(CIFAR10SSL, self).__init__(root=root_path,
                                         train=train,
                                         download=True,
                                         transform=transform,
                                         target_transform=target_transform)

        self.__order_data()
        self.data_range = data_range
        self.__resolve_data_range()
        self.shuffle_data()

    def shuffle_data(self):
        """
        Shuffle both data and its targets randomly.
        """

        # obtain random permutation
        permutation = np.random.permutation(len(self))

        # apply permutation to data and targets
        self.targets = self.targets[permutation]
        self.data = self.data[permutation]

    def mean_and_std(self):
        """
        :return:
            mean and standard deviation of dataset per channel.
        """

        return ((self.data/255).mean(axis=(0, 1, 2)),
                (self.data/255).std(axis=(0, 1, 2)))

    def __order_data(self):
        """
        Order data sequentially by class index (0 to 9).
        """

        # get order from targets then apply it to data and targets
        order = np.asarray(self.targets).argsort()
        self.targets = np.asarray(self.targets)[order].astype(np.int64)
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

        # check if data_range is within boundaries and valid
        max_val = 5000 if self.train else 1000
        assert 0 <= range_l < range_h <= max_val, f'invalid range (max {max_val} img/class)'

        # select data and targets specified by validated data_range
        ids = [idx + max_val * nclass for nclass in range(len(self.classes)) for idx in range(range_l, range_h)]
        self.data = self.data[ids]
        self.targets = self.targets[ids]


if __name__ == "__main__":
    cifar10ds = CIFAR10SSL(root_path='./cifar10', train=True, data_range=(0, 5000))
    print('Cantidad de datos en dataset: ', len(cifar10ds))
    mean, std = cifar10ds.mean_and_std()
    print(f"""
    Mean +/- Std: {mean} +/- {std}
    """)
