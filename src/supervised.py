import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.cifar10 import CIFAR10SSL
from models.wide_resnet import WideResNet
from train.baseline import ModelTrainer

if __name__ == '__main__':
    # define model Wide ResNet
    wrn_model = WideResNet(depth=28,
                           width=2,
                           n_classes=10,
                           dropout=0.3)
    wrn_model.to('cuda')

    # train transformations with mild data augmentation
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                              (0.24703223, 0.24348513, 0.26158784)),
         transforms.RandomHorizontalFlip()])

    # test and val transformations, no data augmentation
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                              (0.24703223, 0.24348513, 0.26158784))])

    # target transformation
    target_transform = lambda label: F.one_hot(torch.tensor(label), num_classes=10).to(torch.float32)

    # construct train, validation and test sets
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

    # construct data loaders for model
    train_dl = DataLoader(dataset=data_train,
                          batch_size=512,
                          shuffle=True)

    val_dl = DataLoader(dataset=data_val,
                        batch_size=100,
                        shuffle=False)

    test_dl = DataLoader(dataset=data_test,
                         batch_size=100,
                         shuffle=False)

    # define training scheme
    trainer = ModelTrainer(model=wrn_model,
                           epochs=5,
                           optimizer='sgd',
                           lr=0.03,
                           train_set=data_train,
                           val_set=data_val,
                           batch_size=128,
                           device='cuda')

    torch.cuda.empty_cache()
    trainer.train()
