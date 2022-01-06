from typing import Tuple

import torch.nn as nn
import torchvision.transforms as transforms

from definitions import CIFAR10_MEAN, CIFAR10_STD, DATASETS_DIR


class RandAugmentTransform(nn.Module):
    """
    RandAugment transformation module.

    A proxy class for RandAugment data augmentation technique
    already implemented in pytorch, but appending ToTensor
    and Normalize transformations for its use in training.
    """

    def __init__(self,
                 N: int = 2,
                 M: int = 9,
                 mean: Tuple[int, ...] = CIFAR10_MEAN,
                 std: Tuple[int, ...] = CIFAR10_STD):
        """
        Constructor of RandAugmentTransform.

        :param N:
            Number of augmentations transformations to apply sequentially.
        :param M:
            Magnitude for all transformations.
        :param mean:
            three-tuple of mean channel-wise for normalization.
        :param std:
            three-tuple of standard deviation channel-wise for normalization.
        """

        super().__init__()

        # transformation params
        self.N = N
        self.M = M
        self.chn_mean = mean
        self.chn_std = std

        # composition of transforms
        self.composition = transforms.Compose([
            transforms.RandAugment(num_ops=self.N, magnitude=self.M),
            transforms.ToTensor(),
            transforms.Normalize(self.chn_mean, self.chn_std)
        ])

    def forward(self, x):
        """
        Forward pass through RandAugmentTransform module.

        :param x:
            input matrix, usually an image.
        :return:
            RandAugmentTransform module output.
        """

        out = self.composition(x)
        return out


if __name__ == "__main__":
    from src.data.cifarssl import CIFAR10SSL

    path = DATASETS_DIR / 'cifar10'
    trans = RandAugmentTransform()
    data = CIFAR10SSL(root_path=path,
                      train=True,
                      data_range=(0, 400),
                      weak_transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
                      ]),
                      strong_transform=trans)

    ex1, ex11, _ = data[0]
