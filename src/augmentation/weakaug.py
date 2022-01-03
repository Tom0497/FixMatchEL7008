import torch.nn as nn
import torchvision.transforms as transforms

CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)

CIFAR100_MEAN = (0.50707516, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733429, 0.25643846, 0.27615047)


class WeakAugmentation(nn.Module):
    """
    Module of weak data augmentation.

    The first component of this module transforms data from numpy array
    or a PIL image to a PyTorch Tensor. It's important to notice that
    this transformation automatically change the data type to float and
    adapts it to the range [0, 1] as documentation specifies. Also, the
    channel order changes from WxHxC to CxWxH, given that PyTorch works
    this way.

    The second component normalize each channel of an image using a
    three-tuple mean and std. By default, CIFAR 10 mean and std are used
    but can be modified in constructor.

    For data augmentation itself, two transformations can be applied,
    random horizontal flip where the probability can be adapted, and
    translation in X and Y direction with a maximum proportion which
    can be also passed as a parameter in constructor.

    If transformation receive a probability of 0 and proportion of 0
    for translation, they are not applied and the transformation serves
    as a data type conversion.
    """

    def __init__(self,
                 h_flip_prob: float = 0.5,
                 translate: tuple[float, float] = (.125, .125),
                 mean: tuple[int, ...] = CIFAR10_MEAN,
                 std: tuple[int, ...] = CIFAR10_STD):
        """
        Constructor of WeakAugmentation module.

        :param h_flip_prob:
            probability of horizontal flip.
        :param translate:
            maximum proportion of shift in X and Y axes.
        :param mean:
            three-tuple of mean channel-wise for normalization.
        :param std:
            three-tuple of standard deviation channel-wise for normalization.
        """

        super().__init__()

        # transformation params
        self.h_flip_prob = h_flip_prob
        self.translation_prop = translate
        self.chn_mean = mean
        self.chn_std = std

        # transformations considered
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.chn_mean, self.chn_std)
        self.random_h_flip = transforms.RandomHorizontalFlip(p=h_flip_prob)
        self.translation = transforms.RandomAffine(degrees=0, translate=translate)

        # list of transformations and composition
        self.transform_list = self.__resolve_transforms()
        self.composition = transforms.Compose(self.transform_list)

    def forward(self, x):
        """
        Forward pass through WeakAugmentation module.

        :param x:
            input matrix, usually an image.
        :return:
            WeakAugmentation module output.
        """

        out = self.composition(x)
        return out

    def __resolve_transforms(self):
        """
        Determine whether to use random transformations.

        :return:
            list of transformations to apply.
        """

        # all data is transformed to tensor and normalized
        transforms_list = [self.to_tensor, self.normalize]

        # random transformations only if parameters are not 0
        if self.h_flip_prob != 0:
            transforms_list.append(self.random_h_flip)
        if self.translation_prop != (0, 0):
            transforms_list.append(self.translation)

        return transforms_list
