import time
from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from train.summary.summary import Summary


class AbstractTrainer(ABC):
    """
    Classification model abstract trainer.

    This abstract class defines the common functionality that
    both supervised and semi-supervised training schemes share.
    In particular, evaluation method is the same for both cases
    given that the difference lays in the training step and
    loss functions each one considers.
    """

    def __init__(self,
                 model: nn.Module,
                 epochs: int,
                 optimizer: str,
                 lr: float,
                 val_set: Dataset,
                 batch_size: int,
                 early_stopping: int,
                 device: str):
        """
        Constructor of AbstractTrainer.

        :param model:
            classification model to be trained.
        :param epochs:
            maximum number of steps to train.
        :param optimizer:
            optimization method, either ADAM or SGD.
        :param lr:
            initial learning rate for optimization method.
        :param val_set:
            validation set, instance of torch.utils.data.Dataset.
        :param batch_size:
            batch size to use in DataLoader for training set.
        :param early_stopping:
            patience in steps for early stopping (0 means no ES).
        :param device:
            either cpu or cuda (gpu).
        """

        # model, steps, learning rate, device (cpu or cuda), early stopping
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.early_stopping = early_stopping

        # loss function (cross-entropy) and optimizer (SGD or ADAM)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = self.__resolve_optimizer(optimizer)

        # batch size and data-loaders for validation set
        self.batch_size = batch_size
        self.val_dl = DataLoader(val_set,
                                 batch_size=100,
                                 shuffle=False)

        # learning rate schedule (cosine decay)
        self.lr_scheduler = self._set_lr_scheduler()

        # log for total training time
        self.train_time = 0
        self.last_epoch = epochs

        # summary for logging of important data
        self.summary = Summary()

    def train(self):
        """
        Execute the training of the model.
        """

        # start time and patience counter
        print('\nBeginning model training\n')
        init_time = time.time()
        patience = 0
        last_val_loss, best_val_loss = 1e5, 1e5

        for epoch in range(1, self.epochs + 1):
            # training stage
            temp_train_loss, train_accuracy, ep_time, it_time = self._train_epoch()

            # evaluation stage
            temp_val_loss, val_accuracy, _ = self.evaluate_model(self.model,
                                                                 self.val_dl,
                                                                 self.device)

            # log and display current epoch metrics
            self.summary.log_epoch(epoch=epoch,
                                   train_loss=temp_train_loss,
                                   val_loss=temp_val_loss,
                                   train_acc=train_accuracy,
                                   val_acc=val_accuracy,
                                   epoch_time=ep_time,
                                   iter_time=it_time)

            # save model with the lowest validation loss
            if best_val_loss > temp_val_loss:
                best_val_loss = temp_val_loss
                best_time = time.time() - init_time
                self.summary.log_model(epoch=epoch,
                                       train_time=best_time,
                                       best=True,
                                       model=self.model)

            # check early stopping conditions
            last_val_loss, patience, terminate = self.__early_stop(actual_loss=temp_val_loss,
                                                                   last_loss=last_val_loss,
                                                                   patience=patience)
            if terminate:
                self.last_epoch = epoch
                print(f'\nTraining early stopped at {epoch} epochs.\n')
                break

        # total training time
        training_time = time.time() - init_time
        self.train_time = training_time
        print(f'\nTraining completed -- training time: {training_time:.2f}s\n')

        # register last state of trained model
        self.summary.log_model(epoch=self.last_epoch,
                               train_time=training_time,
                               best=False,
                               model=self.model)

    def __resolve_optimizer(self, optimizer: str):
        """
        Choose an optimizer based on string input.

        Choices for optimizers are ADAM (Adaptive Moment Estimation) or
        SGD (Stochastic Gradient Descent). The user can decide only the
        initial learning rate, but all other parameters of these optimizers
        are fixed. <https://arxiv.org/pdf/1605.07146.pdf> gives the best
        values obtained through extensive experimentation.

        :return:
            optimizer using string name. Options -> [adam, sgd].
        """

        assert isinstance(optimizer, str), 'optimizer name must be string'
        assert optimizer.lower() in ('adam', 'sgd'), 'only ADAM and SGD supported'

        # weight_decay based on whether data is CIFAR 10 or 100
        weight_decay = 0.0005 if self.model.n_classes == 10 else 0.001
        if optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(),
                              lr=self.lr,
                              betas=(0.9, 0.999),
                              weight_decay=weight_decay)

        return optim.SGD(self.model.parameters(),
                         lr=self.lr,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=weight_decay)

    @abstractmethod
    def _train_epoch(self):
        """
        Protected method - Perform one epoch of training.
        """

        return NotImplementedError

    @abstractmethod
    def _set_lr_scheduler(self):
        """
        Protected method - Configures cosine decay for learning rate.
        """

        return NotImplementedError

    def __early_stop(self,
                     actual_loss: float,
                     last_loss: float,
                     patience: int):
        """
        Private method - Determine if early stopping conditions are met.

        :param actual_loss:
            validation loss of actual training epoch.
        :param last_loss:
            validation loss of previous epoch.
        :param patience:
            number of steps validation loss has not improved.

        :return:
            actual loss, patience and whether to terminate training.
        """

        # validation loss doesn't improve
        if actual_loss >= last_loss:
            patience += 1
        # patience reset when validation loss improves
        else:
            patience = 0

        return (actual_loss,
                patience,
                patience >= self.early_stopping)

    @staticmethod
    def evaluate_model(model: nn.Module,
                       dataloader: DataLoader,
                       device: str):
        """
        Static Method - Evaluate a model in terms of accuracy and CE loss over a dataloader.

        :param model:
            classification model to be evaluated, instance or subclass of torch.nn.Module.
        :param dataloader:
            instance of torch.utils.data.DataLoader, representing dataset for evaluation.
        :param device:
            either cuda or cpu.

        :return:
            accuracy and CE-loss and tuple of predicted vs expected labels.
        """

        # evaluation stage (no-dropout and BN with current batch only), no gradient computation
        model.eval()

        # list to save predicted and expected labels
        predicted, expected = [], []
        temp_loss = 0.0

        with torch.no_grad():
            # iterate over dataloader
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # forward model and get predictions
                outputs = model(inputs)
                predictions = outputs.detach().cpu().numpy().argmax(axis=1)

                # accumulate loss then move labels to cpu
                temp_loss += F.cross_entropy(outputs, labels).item()
                labels = labels.cpu().numpy().argmax(axis=1)

                # save both expected as predicted labels
                predicted.extend(predictions)
                expected.extend(labels)

        # compute CE loss and accuracy
        temp_loss /= len(dataloader)
        val_accuracy = accuracy_score(expected, predicted)

        return (temp_loss,
                val_accuracy,
                (predicted, expected))
