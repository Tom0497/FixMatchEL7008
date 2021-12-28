import torch

from augmentation.weakaug import WeakAugmentation
from data.cifar10 import CIFAR10SSL
from models.wide_resnet import WideResNet
from train.baseline import ModelTrainer
from utils import one_hot_int


def options():

    return 0


if __name__ == '__main__':
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
    transform_train = WeakAugmentation()
    transform_test = WeakAugmentation(h_flip_prob=0, translate=(0, 0))
    target_transform = one_hot_int()

    # train, validation and test sets
    data_train = CIFAR10SSL(root_path='./cifar10',
                            train=True,
                            data_range=(0, 4000),
                            transform=transform_train,
                            target_transform=target_transform)
    data_val = CIFAR10SSL(root_path='./cifar10',
                          train=True,
                          data_range=(4000, 5000),
                          transform=transform_test,
                          target_transform=target_transform)
    data_test = CIFAR10SSL(root_path='./cifar10',
                           train=False,
                           data_range=(0, 1000),
                           transform=transform_test,
                           target_transform=target_transform)

    # training scheme
    trainer = ModelTrainer(model=wrn_model,
                           epochs=5,
                           optimizer='sgd',
                           lr=0.03,
                           train_set=data_train,
                           val_set=data_val,
                           batch_size=128,
                           early_stopping=15,
                           device='cuda')

    # empty gpu cache before train model
    torch.cuda.empty_cache()
    trainer.train()
