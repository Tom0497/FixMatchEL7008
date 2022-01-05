from pathlib import Path
from typing import Tuple, Optional, Callable, Any

import numpy as np
from torchvision.datasets import CIFAR10

from definitions import DATASETS_DIR
from PIL import Image


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

    # number of examples per class in train and test
    per_class_train = 5000
    per_class_test = 1000

    def __init__(self,
                 root_path: str or Path,
                 train: bool,
                 data_range: Tuple[int, int],
                 weak_transform: Optional[Callable] = None,
                 strong_transform: Optional[Callable] = None,
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
                                         target_transform=target_transform)

        self.__order_data()
        self.data_range = data_range
        self.__resolve_data_range()
        self.shuffle_data()

        # weak and strong transformations
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        """
        Get example from dataset using index.

        :param index:
            index to search in database.

        :return:
            Image applying transformations if exists, and label.
        """

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        weak, strong = None, None

        # img contains weak augmentation
        if self.weak_transform is not None:
            weak = self.weak_transform(img)
        # strong contains strong augmentation
        if self.strong_transform is not None:
            strong = self.strong_transform(img)

        # target transform
        if self.target_transform is not None:
            target = self.target_transform(target)

        weak = img if not self.weak_transform else weak
        # if not strong augmentation, usual return
        if not self.strong_transform:
            return weak, target

        return weak, strong, target

    def shuffle_data(self):
        """
        Shuffle both data and its targets randomly.
        """

        # obtain random permutation
        permutation = np.random.permutation(len(self))

        # apply permutation to data and targets
        self.targets = self.targets[permutation]
        self.data = self.data[permutation]

    def mean_and_std(self) -> Tuple[list, list]:
        """
        :return:
            mean and standard deviation of dataset per channel.
        """

        return ((self.data / 255).mean(axis=(0, 1, 2)),
                (self.data / 255).std(axis=(0, 1, 2)))

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
        max_val = self.per_class_train if self.train else self.per_class_test
        assert 0 <= range_l < range_h <= max_val, f'invalid range (max {max_val} img/class)'

        # select data and targets specified by validated data_range
        ids = [idx + max_val * nclass for nclass in range(len(self.classes)) for idx in range(range_l, range_h)]
        self.data = self.data[ids]
        self.targets = self.targets[ids]


class CIFAR100SSL(CIFAR10SSL):
    """
    Dataset CIFAR100 adapted for semi-supervised learning.

    Adapts CIFAR100 dataset class provided by PyTorch by allowing a
    range which selects a subset of images for each class.

    CIFAR-100 dataset consists of 60k 32x32 colour images, from 100
    classes, 600 images per class. 50k are training images and 10k
    are test images. The home page of the dataset is provided
    in <https://www.cs.toronto.edu/~kriz/cifar.html> by the
    University of Toronto.

    This is a subclass of the `CIFAR10SSL` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    # number of examples per class in train and test
    per_class_train = 500
    per_class_test = 100


if __name__ == "__main__":
    path = DATASETS_DIR / 'cifar10'
    cifar10ds = CIFAR10SSL(root_path=path,
                           train=True,
                           data_range=(0, 5000))
    print('Total number of images in dataset: ', len(cifar10ds))
    print('Total number of classes in dataset: ', len(cifar10ds.classes))
    mean, std = cifar10ds.mean_and_std()
    print(f"""
    Mean +/- Std: {mean} +/- {std}
    """)

    path1 = DATASETS_DIR / 'cifar100'
    cifar100ds = CIFAR100SSL(root_path=path1,
                             train=True,
                             data_range=(0, 500))
    print('Total number of images in dataset: ', len(cifar100ds))
    print('Total number of classes in dataset: ', len(cifar100ds.classes))
    mean, std = cifar100ds.mean_and_std()
    print(f"""
    Mean +/- Std: {mean} +/- {std}
    """)
