from pathlib import Path
from textwrap import dedent
from typing import Tuple
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from definitions import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from definitions import RUNS_DIR, DATASETS_DIR
from src.augmentation.weakaug import WeakAugmentation
from src.data.cifarssl import CIFAR10SSL, CIFAR100SSL
from src.models.wide_resnet import WideResNet
from src.train.supervisedtrainer import SupervisedTrainer
from src.utils import one_hot_int


class Plotter:
    """
    Construct figures with main results of training.
    """

    def __init__(self,
                 base_path: str or Path,
                 n_classes: int):
        """
        Constructor of Plotter.

        :param base_path:
            path where the logs from training should be.
        :param n_classes:
            classes of dataset, expected 10 or 100.
        """

        # main paths
        self.base_path = base_path
        self.metrics_path = base_path / 'metrics'
        self.models_path = base_path / 'models'

        # dicts
        self.epoch_logs = torch.load(f=self.metrics_path / 'epoch_logs.pt')
        self.step_logs = torch.load(f=self.metrics_path / 'step_logs.pt')
        self.hyper = torch.load(f=self.models_path / 'hyperparams.pt')

        # model loading
        self.best = torch.load(f=self.models_path / 'best.pt')
        self.last = torch.load(f=self.models_path / 'last.pt')
        self.depth, self.width = self.__infer_model()
        self.n_classes = n_classes
        self.model = WideResNet(depth=self.depth,
                                width=self.width,
                                n_classes=self.n_classes)
        self.model.load_state_dict(self.best['model_state_dict'])
        self.short_name = self.hyper['model'].replace(" - WRN-", "-")
        self.data_name = 'CIFAR-10' if self.n_classes == 10 else 'CIFAR-100'

    def print_metadata(self, fixmatch: bool = False):

        model_meta = dedent(f"""
        Model - {self.short_name} - {self.hyper['n_params']} parameters - {self.data_name}

        Epochs (Max) - {self.hyper['epochs']}
        Batch size - {self.hyper['batch_size']}
        Initial lr -  {self.hyper['lr']}
        Early stopping - {self.hyper['es']} patience epochs 

        Images range per class:""")

        if fixmatch:
            data_l = (self.hyper['train_range'][1] - self.hyper['train_range'][0]) * self.n_classes
            data_ul = (self.hyper['unlabeled_range'][1] - self.hyper['unlabeled_range'][0]) * self.n_classes
            batches_l = ceil(data_l / self.hyper['batch_size'])
            batches_ul = ceil(data_ul / (self.hyper['batch_size'] * self.hyper['mu']))

            factor = max(batches_l, batches_ul)

            model_meta += dedent(f"""
                Labeled - {self.hyper['train_range']}
                Unlabeled - {self.hyper['unlabeled_range']}
                Validation - {self.hyper['val_range']}
            
            FixMatch Hyper-params
                lambda_u - {self.hyper['lambda_u']}
                tau - {self.hyper['tau']}
                mu - {self.hyper['mu']}
                
            Impurity (best model) -- {self.epoch_logs['train_impurity'][self.best['epoch']-1] / factor:.4f}
            Mask Rate (best model) -- {self.epoch_logs['train_mask_rate'][self.best['epoch']-1] / factor:.4f}
            """)
        else:
            model_meta += dedent(f"""
            Labeled - {self.hyper['train_range']}
            Validation - {self.hyper['val_range']}
            """)

        model_meta += dedent(f"""
        Best model in val loss:
            Training time - {self.best['train_time']:.2f}s
            Epoch - {self.best['epoch']}
            train_loss: {self.best['train_loss']:.4f} | train_acc: {self.best['train_acc']:.4f}
            val_loss: {self.best['val_loss']:.4f} | val_acc: {self.best['val_acc']:.4f}
            
        Total training time - {self.last['train_time']:.2f}s - Epochs - {self.last['epoch']}
        """)

        print(dedent(model_meta))

    def plot_loss(self, save: str or Path = None):

        train_loss = self.epoch_logs['train_loss']
        val_loss = self.epoch_logs['val_loss']
        model_name = self.short_name
        data_name = self.data_name

        self.plot_pair(first=train_loss,
                       second=val_loss,
                       labels=('train', 'validation'),
                       axs_labels=('epoch', None),
                       title=f'Training loss - {model_name} - {data_name}',
                       save=save)

    def plot_accu(self, save: str or Path = None):

        train_accu = self.epoch_logs['train_accuracy']
        val_accu = self.epoch_logs['val_accuracy']
        model_name = self.short_name
        data_name = self.data_name

        self.plot_pair(first=train_accu,
                       second=val_accu,
                       labels=('train', 'validation'),
                       axs_labels=('epoch', None),
                       title=f'Training accuracy - {model_name} - {data_name}',
                       save=save)

    def plot_ssl_metrics(self, save: str or Path = None):

        impurity = self.epoch_logs['train_impurity']
        mask_rate = self.epoch_logs['train_mask_rate']
        model_name = self.short_name
        data_name = self.data_name

        data_l = (self.hyper['train_range'][1] - self.hyper['train_range'][0]) * self.n_classes
        data_ul = (self.hyper['unlabeled_range'][1] - self.hyper['unlabeled_range'][0]) * self.n_classes
        batches_l = ceil(data_l / self.hyper['batch_size'])
        batches_ul = ceil(data_ul / (self.hyper['batch_size'] * self.hyper['mu']))

        factor = max(batches_l, batches_ul)

        self.plot_pair(first=np.asarray(impurity) / factor,
                       second=np.asarray(mask_rate) / factor,
                       labels=('impurity', 'mask rate'),
                       axs_labels=('epoch', None),
                       title=f'Impurity | Mask Rate - {model_name} - {data_name}',
                       save=save)

    def plot_ssl_losses(self, save: str or Path = None):

        loss_un = self.epoch_logs['train_un_loss']
        loss_su = self.epoch_logs['train_su_loss']
        model_name = self.short_name
        data_name = self.data_name

        data_l = (self.hyper['train_range'][1] - self.hyper['train_range'][0]) * self.n_classes
        data_ul = (self.hyper['unlabeled_range'][1] - self.hyper['unlabeled_range'][0]) * self.n_classes
        batches_l = ceil(data_l / self.hyper['batch_size'])
        batches_ul = ceil(data_ul / (self.hyper['batch_size'] * self.hyper['mu']))

        factor = max(batches_l, batches_ul)

        self.plot_pair(first=np.asarray(loss_un) / factor,
                       second=np.asarray(loss_su) / factor,
                       labels=('unsupervised', 'supervised'),
                       axs_labels=('epoch', None),
                       title=f'Loss components - {model_name} - {data_name}',
                       save=save)

    @staticmethod
    def plot_pair(first: list,
                  second: list,
                  labels: tuple = ('', ''),
                  axs_labels: tuple = None,
                  title: str = '',
                  ax: plt.Axes = None,
                  save: str or Path = None):

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(first, label=labels[0])
        ax.plot(second, label=labels[1])
        ax.set_title(title)

        if axs_labels:
            ax.set_xlabel(axs_labels[0])
            ax.set_ylabel(axs_labels[1])

        ax.legend()

        if save:
            plt.savefig(save, dpi=300)

    def test_best(self, save: str or Path = None):

        target_transform = one_hot_int(num_classes=self.n_classes)
        if self.n_classes == 10:
            data_path = DATASETS_DIR / 'cifar10'
            transform_test = WeakAugmentation(mean=CIFAR10_MEAN,
                                              std=CIFAR10_STD,
                                              h_flip_prob=0,
                                              translate=(0, 0))
            test_set = CIFAR10SSL(root_path=data_path,
                                  train=False,
                                  data_range=(0, 1000),
                                  weak_transform=transform_test,
                                  target_transform=target_transform)
        else:
            data_path = DATASETS_DIR / 'cifar100'
            transform_test = WeakAugmentation(mean=CIFAR100_MEAN,
                                              std=CIFAR100_STD,
                                              h_flip_prob=0,
                                              translate=(0, 0))
            test_set = CIFAR100SSL(root_path=data_path,
                                   train=False,
                                   data_range=(0, 100),
                                   weak_transform=transform_test,
                                   target_transform=target_transform)

        test_dl = DataLoader(test_set, batch_size=100, shuffle=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        loss, accu, (pred, expec) = SupervisedTrainer.evaluate_model(model=self.model,
                                                                     dataloader=test_dl,
                                                                     device=device)
        print(f'Test -- loss: {loss:.4f} | acc: {accu:.4f}')

        if save:
            cls_text = test_set.classes

            fig, ax = plt.subplots()
            if self.n_classes == 10:
                ConfusionMatrixDisplay.from_predictions(expec, pred,
                                                        display_labels=cls_text,
                                                        ax=ax,
                                                        normalize='true',
                                                        xticks_rotation='vertical',
                                                        values_format='.2f')
            else:
                ConfusionMatrixDisplay.from_predictions(expec, pred,
                                                        ax=ax,
                                                        include_values=False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            model_name = self.short_name
            data_name = self.data_name
            ax.set_title(f'Confusion matrix - {model_name} - {data_name}')
            plt.tight_layout()
            plt.savefig(save, dpi=300)

    def __infer_model(self) -> Tuple[int, int]:

        model_name: str = self.hyper['model']
        abbr = model_name[model_name.index('WRN-') + 4:]

        depth, width = abbr.split('-')
        return int(depth), int(width)


if __name__ == "__main__":
    folder = 'fixmatch1'
    path = RUNS_DIR / folder
    num_classes = 100 if folder in ['baseline4', 'fixmatch4'] else 10
    plttr = Plotter(base_path=path, n_classes=num_classes)

    plttr.print_metadata(fixmatch='fixmatch' in folder)

    plttr.plot_loss(save=path / 'train_loss.png')
    plttr.plot_accu(save=path / 'train_accu.png')
    plttr.test_best(save=path / 'conf_matrix.png')

    if 'fixmatch' in folder:
        plttr.plot_ssl_metrics(save=path / 'ssl_metrics.png')
        plttr.plot_ssl_losses(save=path / 'ssl_losses.png')
