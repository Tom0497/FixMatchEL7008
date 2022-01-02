import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from train.abstract_trainer import AbstractTrainer


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
        :param early_stopping:
            patience in epochs for early stopping (0 means no ES).
        :param device:
            either cpu or cuda (gpu).
        """

        super().__init__(model=model,
                         epochs=epochs,
                         optimizer=optimizer,
                         lr=lr,
                         val_set=val_set,
                         batch_size=batch_size,
                         early_stopping=early_stopping,
                         device=device)

        # data-loader for training set
        self.train_dl = DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True)

    def _train_epoch(self):
        """
        Perform one epoch of supervised training.

        An epoch of supervised training consists of iterating through the
        entire train dataloader and, for each batch of data, do forward
        propagation step, the backpropagation step and update the weights
        of the model.

        :return:
            CE-loss and accuracy computed over training dataset.
        """

        # training stage
        self.model.train()

        # list to save predicted and expected labels
        temp_loss = 0.0
        predicted, expected = [], []

        # for cycle defines an epoch for train set
        for inputs, labels in self.train_dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # set gradient values to zero
            self.optimizer.zero_grad()

            # model forward, loss and gradient computation, weights update
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
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

        return temp_loss, train_accuracy
