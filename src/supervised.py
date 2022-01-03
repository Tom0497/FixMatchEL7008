import argparse

import torch

from definitions import DATASETS_DIR, RUNS_DIR
from src.augmentation.weakaug import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from src.augmentation.weakaug import WeakAugmentation
from src.data.cifarssl import CIFAR10SSL
from src.models.wide_resnet import WideResNet
from src.train.supervisedtrainer import SupervisedTrainer
from src.utils import one_hot_int


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

    return parser.parse_args()


def main(opt: argparse.Namespace) -> int:
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

    model_depth, model_width = opt.model_depth, opt.model_width
    dataset = opt.data
    batch_size = opt.batch_size
    epochs = opt.epochs
    patience = opt.early_stopping
    results_folder = opt.results
    train_range = tuple(opt.train_range)
    val_range = tuple(opt.val_range)

    if dataset == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        n_classes = 10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        n_classes = 100

    # model
    wrn_model = WideResNet(depth=model_depth,
                           width=model_width,
                           n_classes=n_classes,
                           dropout=0.3).to('cuda')

    # transformations
    transform_train = WeakAugmentation(mean=mean,
                                       std=std)
    transform_test = WeakAugmentation(mean=mean,
                                      std=std,
                                      h_flip_prob=0,
                                      translate=(0, 0))
    target_transform = one_hot_int(num_classes=n_classes)

    # train, validation and test sets
    data_path = DATASETS_DIR / dataset
    data_train = CIFAR10SSL(root_path=data_path,
                            train=True,
                            data_range=train_range,
                            transform=transform_train,
                            target_transform=target_transform)
    data_val = CIFAR10SSL(root_path=data_path,
                          train=True,
                          data_range=val_range,
                          transform=transform_test,
                          target_transform=target_transform)

    # training scheme
    trainer = SupervisedTrainer(model=wrn_model,
                                epochs=epochs,
                                optimizer='sgd',
                                lr=0.03,
                                train_set=data_train,
                                val_set=data_val,
                                batch_size=batch_size,
                                early_stopping=patience,
                                device='cuda')

    # empty gpu cache before train model
    torch.cuda.empty_cache()
    trainer.train()

    # save results
    logs_path = RUNS_DIR / results_folder
    trainer.save_logs(base_path=logs_path)

    return 1


if __name__ == '__main__':
    opts = options()
    res = main(opt=opts)
