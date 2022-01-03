import argparse

import torch

from augmentation.weakaug import CIFAR10_MEAN, CIFAR10_STD
from augmentation.weakaug import WeakAugmentation
from data.cifarssl import CIFAR10SSL
from definitions import DATASETS_DIR, RUNS_DIR
from models.wide_resnet import WideResNet
from train.supervisedtrainer import SupervisedTrainer
from utils import one_hot_int


def options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="dataset for training",
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')
    parser.add_argument("-e", "--epochs",
                        help="number of epochs for training",
                        type=int,
                        default=30)

    return parser.parse_args()


def main(opt=None) -> int:
    """
    Fully supervised classification over CIFAR 10.

    A Wide ResNet model of depth 28 and width 2 is trained for
    classification over the dataset CIFAR 10. This serves as a
    baseline for semi-supervised learning (SSL) task.

    CIFAR consists of 60000 images for training and 10000 for
    test, in order to have a validation set, 10000 images from
    training set are destined for validation.

    All data is normalized using mean and standard deviation
    from training set. Data augmentation composed of random
    horizontal flip and random translations in X and Y are
    used for training. Labels are transformed to one hot encoding.
    """

    # model
    wrn_model = WideResNet(depth=28,
                           width=2,
                           n_classes=10,
                           dropout=0.3).to('cuda')

    # transformations
    transform_train = WeakAugmentation(mean=CIFAR10_MEAN,
                                       std=CIFAR10_STD)
    transform_test = WeakAugmentation(mean=CIFAR10_MEAN,
                                      std=CIFAR10_STD,
                                      h_flip_prob=0,
                                      translate=(0, 0))
    target_transform = one_hot_int()

    # train, validation and test sets
    data_path = DATASETS_DIR / 'cifar10'
    data_train = CIFAR10SSL(root_path=data_path,
                            train=True,
                            data_range=(0, 25),
                            transform=transform_train,
                            target_transform=target_transform)
    data_val = CIFAR10SSL(root_path=data_path,
                          train=True,
                          data_range=(4000, 4100),
                          transform=transform_test,
                          target_transform=target_transform)

    # training scheme
    trainer = SupervisedTrainer(model=wrn_model,
                                epochs=5,
                                optimizer='sgd',
                                lr=0.03,
                                train_set=data_train,
                                val_set=data_val,
                                batch_size=512,
                                early_stopping=15,
                                device='cuda')

    # empty gpu cache before train model
    torch.cuda.empty_cache()
    trainer.train()

    # save results
    logs_path = RUNS_DIR / 'alldata'
    trainer.save_logs(base_path=logs_path)

    return 1


if __name__ == '__main__':
    # opts = options()
    res = main()
