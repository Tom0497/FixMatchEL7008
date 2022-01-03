import sys

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from src.train.abstract_trainer import AbstractTrainer
from src.utils import cosine_decay


class SupervisedTrainer(AbstractTrainer):
    """
    Classification model supervised training.

    This class inherits from AbstractTrainer and represents a supervised
    training scheme, i.e. all data used for training has a known label.
    At least the _train_epoch abstract method must be implemented, given
    that this step differentiates a fully supervised training from a semi
    supervised training.

    Logs are used to register accuracy and loss progression in every
    epoch of training. Also, the best model is saved using the loss
    on validation set as the criterion.
    """

    def __init__(self,
                 model: nn.Module,
                 epochs: int,
                 optimizer: str,
                 lr: float,
                 train_set: Dataset,
                 val_set: Dataset,
                 batch_size: int,
                 early_stopping: int,
                 device: str):
        """
        Constructor of SupervisedTrainer.

        :param model:
            classification model to be trained.
        :param epochs:
            maximum number of steps to train.
        :param optimizer:
            optimization method, either ADAM or SGD.
        :param lr:
            initial learning rate for optimization method.
        :param train_set:
            training set, instance of torch.utils.data.Dataset.
        :param val_set:
            validation set, instance of torch.utils.data.Dataset.
        :param batch_size:
            batch size to use in DataLoader for training set.
        :param early_stopping:
            patience in steps for early stopping (0 means no ES).
        :param device:
            either cpu or cuda (gpu).
        """

        # data-loader for training set
        self.train_dl = DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True)

        super().__init__(model=model,
                         epochs=epochs,
                         optimizer=optimizer,
                         lr=lr,
                         val_set=val_set,
                         batch_size=batch_size,
                         early_stopping=early_stopping,
                         device=device)

    def _train_epoch(self):
        """
        Protected method - Perform one epoch of supervised training.

        An epoch of supervised training consists of iterating through the
        entire train dataloader and, for each batch of data, do forward
        propagation step, the backpropagation step and update the weights
        of the model.

        :return:
            CE-loss and accuracy computed over training dataset.
        """

        # training stage
        self.model.train()

        # placeholders for loss and accuracy computation
        train_loss = 0
        train_accuracy = 0

        # number of training images
        temp_count = 0
        image_num = len(self.train_dl.dataset.data)

        # for cycle defines an epoch for train set
        for inputs, labels in self.train_dl:
            temp_count += len(labels)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # set gradient values to zero
            self.optimizer.zero_grad()

            # model forward, loss and gradient computation, weights update
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # learning rate schedule
            self.lr_scheduler.step()

            # predictions for accuracy computation
            predictions = outputs.detach().cpu().numpy().argmax(axis=1)
            labels = labels.cpu().numpy().argmax(axis=1)

            # loss and accuracy over a single step of training
            temp_loss = loss.item()
            temp_accu = accuracy_score(labels, predictions)
            sys.stdout.write(f'\r[{temp_count}/{image_num}] loss: {temp_loss:.4f} | accu: {temp_accu:.4f}')

            # accumulated loss for epoch
            train_loss += temp_loss
            train_accuracy += temp_accu

        # accuracy and loss computation for one epoch of training
        train_accuracy /= len(self.train_dl)
        train_loss /= len(self.train_dl)

        return train_loss, train_accuracy

    def _set_lr_scheduler(self) -> optim.lr_scheduler:
        """
        Protected method - Configures cosine decay for learning rate.

        :return:
            cosine decay learning rate scheduler.
        """

        max_steps = self.epochs * len(self.train_dl)

        return optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                           lr_lambda=cosine_decay(steps=max_steps))
