import argparse

import torch

from definitions import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from definitions import DATASETS_DIR, RUNS_DIR
from src.augmentation.weakaug import WeakAugmentation
from src.data.cifarssl import CIFAR10SSL, CIFAR100SSL
from src.models.wide_resnet import WideResNet
from src.train.supervisedtrainer import SupervisedTrainer
from src.utils import one_hot_int, options


def main(opt: argparse.Namespace) -> int:
    """
    Fully supervised classification over CIFAR dataset (10 or 100).

    A Wide ResNet model of adjustable depth and width is trained for
    classification over the dataset CIFAR. This serves as a
    baseline for semi-supervised learning (SSL) task.

    CIFAR consists of 60000 images for training and 10000 for
    test, in order to have a validation set, 10000 images from
    training set are destined for validation.

    All data is normalized using mean and standard deviation
    from training set. Data augmentation composed of random
    horizontal flip and random translations in X and Y are
    used for training. Labels are transformed to one hot encoding.
    """

    # wide resnet width and depth
    model_depth, model_width = opt.model_depth, opt.model_width

    # dataset to use (cifar10 or cifar100)
    dataset = opt.data

    # trainer hyper parameters
    batch_size = opt.batch_size
    epochs = opt.epochs
    patience = opt.early_stopping
    results_folder = opt.results

    # labeled and validation ranges per class
    train_range = tuple(opt.train_range)
    val_range = tuple(opt.val_range)

    # which dataset to use
    if dataset == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        n_classes = 10
        DataClass = CIFAR10SSL
    else:  # cifar 100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        n_classes = 100
        DataClass = CIFAR100SSL

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrn_model = WideResNet(depth=model_depth,
                           width=model_width,
                           n_classes=n_classes,
                           dropout=0.3).to(device)

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
    data_train = DataClass(root_path=data_path,
                           train=True,
                           data_range=train_range,
                           weak_transform=transform_train,
                           target_transform=target_transform)
    data_val = DataClass(root_path=data_path,
                         train=True,
                         data_range=val_range,
                         weak_transform=transform_test,
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
                                device=device)

    # empty gpu cache before train model
    torch.cuda.empty_cache()
    trainer.train()

    # save results
    logs_path = RUNS_DIR / results_folder
    trainer.summary.save_logs(base_path=logs_path)

    return 1


if __name__ == '__main__':
    opts = options()
    res = main(opt=opts)
