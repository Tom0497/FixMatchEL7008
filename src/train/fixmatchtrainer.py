import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

from src.train.abstract_trainer import AbstractTrainer
from train.summary.summaryssl import SummarySSL
from src.utils import cosine_decay


class FixMatchTrainer(AbstractTrainer):
    """
    Classification model semi supervised trainer.

    This class inherits from AbstractTrainer and represents a semi supervised
    training scheme, i.e. part of the data is labeled but hte major part of it
    is unlabeled, and to integrate it to the training, FixMatch is implemented.

    At least the _train_epoch abstract method must be implemented, given
    that this step differentiates a semi supervised training from a fully
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
                 labeled_set: Dataset,
                 unlabeled_set: Dataset,
                 val_set: Dataset,
                 batch_size: int,
                 mu: float,
                 tau: float,
                 lambda_u: float,
                 early_stopping: int,
                 device: str):
        """
        Constructor of FixMatchTrainer.

        :param model:
            classification model to be trained.
        :param epochs:
            maximum number of steps to train.
        :param optimizer:
            optimization method, either ADAM or SGD.
        :param lr:
            initial learning rate for optimization method.
        :param labeled_set:
            labeled set, instance of torch.utils.data.Dataset.
        :param unlabeled_set:
            unlabeled set, instance of torch.utils.data.Dataset.
        :param val_set:
            validation set, instance of torch.utils.data.Dataset.
        :param mu:
            batch size multiplier in terms of labeled batch size.
        :param tau:
            threshold for which a pseudo label is considered valid.
        :param lambda_u:
            value that balances between supervised and unsupervised loss.
        :param batch_size:
            batch size to use in DataLoader for training set.
        :param early_stopping:
            patience in steps for early stopping (0 means no ES).
        :param device:
            either cpu or cuda (gpu).
        """

        # mu for FixMatch and batch size for unlabeled data.
        self.mu = mu
        self.tau = tau
        self.lambda_u = lambda_u
        self.batch_size_un = int(batch_size * mu)

        # data-loader for labeled and unlabeled sets
        self.labeled_dl = DataLoader(labeled_set,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     pin_memory=True if device == 'cuda' else False)
        self.unlabeled_dl = DataLoader(unlabeled_set,
                                       batch_size=self.batch_size_un,
                                       shuffle=True,
                                       pin_memory=True if device == 'cuda' else False)

        self.max_steps = max(len(self.labeled_dl), len(self.unlabeled_dl))

        self.probabilities = nn.Softmax(dim=-1)
        self.un_loss_func = nn.CrossEntropyLoss(reduction='none')

        super().__init__(model=model,
                         epochs=epochs,
                         optimizer=optimizer,
                         lr=lr,
                         val_set=val_set,
                         batch_size=batch_size,
                         early_stopping=early_stopping,
                         device=device)

        # register training hyperparameters
        self.summary = SummarySSL()
        self.summary.log_hyperparams(hyperparams=self.get_hyperparams())

    def _set_lr_scheduler(self):
        """
        Protected method - Configures cosine decay for learning rate.

        :return:
            cosine decay learning rate scheduler.
        """

        max_steps = self.epochs * self.max_steps

        return optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                           lr_lambda=cosine_decay(steps=max_steps))

    def _train_epoch(self):
        """
        Protected method - Perform one epoch of semi supervised training.

        An epoch of semi supervised training consists of iterating through
        both a labeled and unlabeled batches of data, forward them through
        the model, and, in this case, under FixMatch scheme compute pseudo
        labels apply a mask and finally compute an unsupervised loss.

        :return:
            CE-loss and accuracy computed over training dataset.
        """

        # training stage
        self.model.train()
        init_time = time.time()
        it_per_sec = 0

        # placeholders for losses, accuracy, impurity and mask rate computation
        train_loss = 0
        train_su_loss = 0
        train_un_loss = 0
        train_accuracy = 0
        train_mask_rate = 0
        train_impurity = 0

        # number of training images
        temp_count = 0

        # data loaders iterators
        labeled_iter = iter(self.labeled_dl)
        unlabeled_iter = iter(self.unlabeled_dl)

        # for cycle defines an epoch for train set
        for i in range(1, self.max_steps + 1):
            try:
                # try to get batch from labeled dataloader
                labeled, labels = next(labeled_iter)
            except StopIteration:
                # reset iterator if already used within an epoch
                labeled_iter = iter(self.labeled_dl)
                labeled, labels = next(labeled_iter)

            try:
                # try to get batch from labeled dataloader
                unlabeled_weak, unlabeled_strong, labels_un = next(unlabeled_iter)
            except StopIteration:
                # reset iterator if already used within an epoch
                unlabeled_iter = iter(self.unlabeled_dl)
                unlabeled_weak, unlabeled_strong, labels_un = next(unlabeled_iter)

            # concatenate all examples then move to device
            inputs = torch.cat((labeled, unlabeled_weak, unlabeled_strong), dim=0)
            inputs, labels, labels_un = inputs.to(self.device), labels.to(self.device), labels_un.to(self.device)

            temp_count += len(labels)
            n_labeled = len(labels)
            n_unlabeled = len(unlabeled_weak)

            # set gradient values to zero
            self.optimizer.zero_grad()

            # model forward and output split
            outputs = self.model(inputs)
            out_l, out_u_w, out_u_s = torch.split(outputs, [n_labeled, n_unlabeled, n_unlabeled], dim=0)

            # distribution for weak unlabeled, pseudo-label and indicator function
            q_b = self.probabilities(out_u_w)
            q_b_max, q_b_hat = torch.max(q_b, dim=-1)
            indicator = torch.ge(q_b_max, self.tau)

            # supervised and unsupervised losses and gradient computation, weights update
            supervised_loss = self.loss_function(out_l, labels)
            unsupervised_loss_sparse = torch.masked_select(self.un_loss_func(out_u_s, q_b_hat), indicator)
            unsupervised_loss = torch.sum(unsupervised_loss_sparse) / n_unlabeled

            # total loss, gradient computation and weights update
            total_loss = supervised_loss + self.lambda_u * unsupervised_loss
            total_loss.backward()
            self.optimizer.step()

            # learning rate schedule
            self.lr_scheduler.step()

            # predictions for accuracy computation
            predictions = out_l.detach().cpu().numpy().argmax(axis=1)
            labels = labels.cpu().numpy().argmax(axis=1)

            # mask rate computation
            indicator_d = indicator.detach().float()
            over_thresh = torch.sum(indicator_d)
            mask_rate = (over_thresh / n_unlabeled).item()

            # impurity computation
            indicator_l = labels_un != q_b_hat.detach()
            impurity = (torch.sum(indicator_d * indicator_l) / over_thresh).item()

            # loss and accuracy over a single step of training
            temp_loss = total_loss.detach().item()
            temp_su_loss = supervised_loss.detach().item()
            temp_un_loss = unsupervised_loss.detach().item()
            temp_accu = accuracy_score(labels, predictions)
            it_per_sec = i / (time.time() - init_time)
            self.summary.log_step_ssl(partial=i,
                                      total=self.max_steps,
                                      loss=temp_loss,
                                      accuracy=temp_accu,
                                      su_loss=temp_su_loss,
                                      un_loss=temp_un_loss,
                                      impurity=impurity,
                                      mask_rate=mask_rate,
                                      it_per_sec=it_per_sec)

            # accumulated loss for epoch
            train_loss += temp_loss
            train_su_loss += temp_su_loss
            train_un_loss += temp_un_loss
            train_accuracy += temp_accu * len(labels)
            train_mask_rate += mask_rate
            train_impurity += impurity

        # log ssl epoch metrics
        self.summary.log_epoch_ssl(train_su_loss=train_su_loss,
                                   train_un_loss=train_un_loss,
                                   train_impurity=train_impurity,
                                   train_mask_rate=train_mask_rate)

        # accuracy and loss computation for one epoch of training
        train_accuracy /= temp_count
        train_loss /= self.max_steps
        total_time = time.time() - init_time

        return train_loss, train_accuracy, total_time, it_per_sec

    def get_hyperparams(self):
        """
        :return:
            Hyper parameters used for training the model.
        """

        hyper_params = {
            'model': str(self.model),
            'n_params': self.model.num_parameters(),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'val_range': self.val_dl.dataset.data_range,
            'train_range': self.labeled_dl.dataset.data_range,
            'es': self.early_stopping,
            'unlabeled_range': self.unlabeled_dl.dataset.data_range,
            'mu': self.mu,
            'tau': self.tau,
            'lambda_u': self.lambda_u
        }
        return hyper_params
