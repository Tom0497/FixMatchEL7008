import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


class ModelTrainer(object):
    """
    Class represents training steps for classification model.

    This class encapsulates the steps of training for a
    classification model, every important variation can be
    achieved through the constructor parameters. It only covers
    supervised learning.

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
                 device: str):
        """
        Constructor of ModelTrainer.

        :param model:
            classification model to be trained.
        :param epochs:
            maximum number of epochs to train.
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
        :param device:
            either cpu or cuda (gpu).
        """

        # model, epochs, learning rate and device (cpu or cuda)
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.device = device

        # loss function (cross-entropy) and optimizer (SGD or ADAM)
        self.lossfun = nn.CrossEntropyLoss()
        self.optimizer = self.__resolve_optimizer(optimizer)

        # logs for accuracy and loss in training and validation sets
        self.train_loss = []
        self.train_accu = []
        self.val_loss = []
        self.val_accu = []

        # batch size and data-loaders for training and validation set
        self.batch_size = batch_size
        self.train_dl = DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True)
        self.val_dl = DataLoader(val_set,
                                 batch_size=100,
                                 shuffle=False)

        # dict for saving of best model based on validation loss
        self.best_model = {'val_loss': 1e5}

        # log for total training time
        self.train_time = 0

    def train(self):
        """
        Execute the training of the model.
        """

        # start time
        init_time = time.time()

        for epoch in range(self.epochs):
            # training stage
            temp_loss = 0.0
            predicted, expected = [], []

            # for cycle defines an epoch for train set
            for inputs, labels in self.train_dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # set gradient values to zero
                self.optimizer.zero_grad()

                # model forward, loss and gradient computation, weights update
                outputs = self.model(inputs)
                loss = self.lossfun(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # predictions for accuracy computation
                predictions = outputs.detach().cpu().numpy().argmax(axis=1)
                labels = labels.cpu().numpy().argmax(axis=1)

                # both labels and predictions are temporally stored to compute epoch accuracy
                predicted.extend(predictions)
                expected.extend(labels)

                # accumulated loss for epoch
                temp_loss += loss.item()

            # accuracy and loss computation for one epoch of training
            train_accuracy = accuracy_score(expected, predicted)
            temp_loss /= len(self.train_dl)

            # log accuracy and loss in training
            self.train_loss.append(temp_loss)
            self.train_accu.append(train_accuracy)

            # evaluation stage (no-dropout and BN with current batch only), no gradient computation
            self.model.eval()
            with torch.no_grad():
                # accuracy and loss in validation set
                temp_val_loss, val_accuracy, _ = self.evaluate_model(self.model,
                                                                     self.val_dl,
                                                                     self.device)

                # log of accuracy and loss on validation set
                self.val_loss.append(temp_val_loss)
                self.val_accu.append(val_accuracy)
            # set model to train mode
            self.model.train()

            print(
                f'[Epoch {epoch}]'
                f' train_loss: {temp_loss:.4f} |'
                f' val_loss: {temp_val_loss:.4f} |'
                f' val_accu: {val_accuracy:.2f} |'
                f' train_accu: {train_accuracy:.2f}')

            # save model with the lowest validation loss
            if self.best_model['val_loss'] > temp_val_loss:
                best_time = time.time() - init_time
                self.best_model = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': temp_loss,
                    'val_loss': temp_val_loss,
                    'train_accu': train_accuracy,
                    'val_accu': val_accuracy,
                    'best_time': best_time
                }

        # total trainig time
        training_time = time.time() - init_time
        self.train_time = training_time
        print(f'Training completed -- training time: {training_time:.3f} s')

    def __resolve_optimizer(self, optimizer: str):
        """
        :return:
            optimizer using string name. Options -> [adam, sgd].
        """

        assert isinstance(optimizer, str) and optimizer in ('adam', 'sgd'), 'only adam or sgd supported'
        if optimizer == 'adam':
            return optim.Adam(self.model.parameters(),
                              lr=self.lr,
                              betas=(0.9, 0.999))
        return optim.SGD(self.model.parameters(),
                         lr=self.lr,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=0.0005)

    def get_logs(self):
        """
        :return:
            accuracy and loss logs during training in train and evaluation sets.
        """

        return (self.train_loss,
                self.train_accu,
                self.val_loss,
                self.val_accu)

    def get_best_model(self):
        """
        :return:
            best model obtained during training based on validation loss.
        """

        return self.best_model

    def get_train_time(self):
        """
        :return:
            training time for total number of epochs (not necessarily best model).
        """

        return self.train_time

    def get_model(self):
        """
        :return:
            model obtained after total number of epochs (not necessarily best model).
        """

        return self.model

    def get_lr(self):
        """
        :return:
            initial learning rate used set for optimizer.
        """

        return self.lr

    def get_optimizer(self):
        """
        :return:
            name of optimizer used for training -> [Adam, SGD].
        """

        if isinstance(self.optimizer, optim.Adam):
            return 'Adam'
        return 'SGD'

    def get_epochs(self):
        """
        :return:
            name of optimizer used for training -> [Adam, SGD].
        """

        return self.epochs

    def get_batch_size(self):
        """
        :return:
            batch size used for training set.
        """

        return self.batch_size

    @staticmethod
    def evaluate_model(model: nn.Module,
                       dataloader: DataLoader,
                       device: str):
        """
        Evaluate a model in terms of accuracy and CE loss over a dataloader.

        :param model:
            classification model to be evaluated, instance or subclass of torch.nn.Module.
        :param dataloader:
            instance of torch.utils.data.DataLoader, representing dataset for evaluation.
        :param device:
            either cuda or cpu.

        :return:
            accuracy and CE-loss and tuple of predicted vs expected labels.
        """

        # list to save predicted and expected labels
        predicted, expected = [], []
        temp_loss = 0.0

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
