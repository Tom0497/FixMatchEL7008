import argparse

import torch

from definitions import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from definitions import DATASETS_DIR, RUNS_DIR
from src.augmentation.strongaug import RandAugmentTransform
from src.augmentation.weakaug import WeakAugmentation
from src.data.cifarssl import CIFAR10SSL, CIFAR100SSL
from src.models.wide_resnet import WideResNet
from src.train.fixmatchtrainer import FixMatchTrainer
from src.utils import one_hot_int, options


def main(opt: argparse.Namespace) -> int:
    """
    Semi Supervised learning with FixMatch over CIFAR (10 or 100).
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
    unlabeled_range = tuple(opt.unlabeled_range)

    # fixmatch parameters
    tau = opt.tau
    mu = opt.mu
    lambda_u = opt.lambda_u

    # randaugment parameters
    N, M = opt.N, opt.M

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
    weak_transform = WeakAugmentation(mean=mean,
                                      std=std)
    strong_transform = RandAugmentTransform(N=N, M=M,
                                            mean=mean,
                                            std=std)
    validation_transforms = WeakAugmentation(mean=mean,
                                             std=std,
                                             h_flip_prob=0,
                                             translate=(0, 0))
    target_transform = one_hot_int(num_classes=n_classes)

    # labeled, unlabeled and validation sets
    data_path = DATASETS_DIR / dataset
    labeled_data = DataClass(root_path=data_path,
                             train=True,
                             data_range=train_range,
                             weak_transform=weak_transform,
                             target_transform=target_transform)
    unlabeled_data = DataClass(root_path=data_path,
                               train=True,
                               data_range=unlabeled_range,
                               weak_transform=weak_transform,
                               strong_transform=strong_transform)
    validation_data = DataClass(root_path=data_path,
                                train=True,
                                data_range=val_range,
                                weak_transform=validation_transforms,
                                target_transform=target_transform)

    # training scheme
    trainer = FixMatchTrainer(model=wrn_model,
                              epochs=epochs,
                              optimizer='sgd',
                              lr=0.03,
                              labeled_set=labeled_data,
                              unlabeled_set=unlabeled_data,
                              val_set=validation_data,
                              batch_size=batch_size,
                              early_stopping=patience,
                              tau=tau,
                              lambda_u=lambda_u,
                              mu=mu,
                              device=device)

    # empty gpu cache before train model
    torch.cuda.empty_cache()
    trainer.train()

    # save results
    logs_path = RUNS_DIR / results_folder
    trainer.summary.save_logs(base_path=logs_path)

    return 1


if __name__ == '__main__':
    opts = options(fixmatch=True)
    res = main(opt=opts)
