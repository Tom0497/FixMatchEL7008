import sys
from pathlib import Path
from textwrap import dedent

import torch
import torch.nn as nn


class Summary:
    """
    Register and logger for a model trainer.

    Offers functionality to a model training by allowing to
    register data and metrics during training, and also registering
    the hyper parameters used by the trainer for the model. It also
    has saving to file functionality.
    """

    def __init__(self):
        """
        Constructor of Summary.
        """

        # logs for epoch and batch training
        self.epoch_logs = dict(
            train_accuracy=[],
            train_loss=[],
            val_accuracy=[],
            val_loss=[]
        )
        self.step_logs = dict(
            accuracy=[],
            loss=[]
        )

        # training hyper parameters
        self.hyperparameters = dict(
            model=None,
            n_params=None,
            epochs=None,
            batch_size=None,
            lr=None,
            val_range=None,
            train_range=None,
            es=None
        )

        # best and last model trained
        self.models = dict(
            best=None,
            last=None
        )

    def log_epoch(self,
                  epoch: int,
                  train_acc: float,
                  train_loss: float,
                  val_acc: float,
                  val_loss: float,
                  epoch_time: float,
                  iter_time: float,
                  log: bool = True):
        """
        Log and display important metrics from an epoch.

        :param epoch:
            training epoch.
        :param train_acc:
            accuracy in training set.
        :param train_loss:
            CE loss in training set.
        :param val_acc:
            accuracy in validation set.
        :param val_loss:
            CE loss in validation set.
        :param epoch_time:
            total time for epoch in seconds.
        :param iter_time:
            average time per iteration in seconds.
        :param log:
            indicate if values are printed for user.
        """

        self.epoch_logs['train_accuracy'].append(train_acc)
        self.epoch_logs['train_loss'].append(train_loss)
        self.epoch_logs['val_accuracy'].append(val_acc)
        self.epoch_logs['val_loss'].append(val_loss)

        if log:
            sys.stdout.write(
                f'\r[Epoch {epoch}] {epoch_time:.1f}s ({iter_time:.2f}it/s)'
                f' train_loss: {train_loss:.4f} |'
                f' train_accu: {train_acc:.4f} |'
                f' val_loss: {val_loss:.4f} |'
                f' val_accu: {val_acc:.4f}\n'
            )

    def log_step(self,
                 partial: int,
                 total: int,
                 loss: float,
                 accuracy: float,
                 it_per_sec: float,
                 log: bool = True):
        """
        Log and display important metrics from a step or batch.

        :param partial:
            number of images already used from training set in current epoch.
        :param it_per_sec:
            iterations per second estimation.
        :param total:
            number of images in training set.
        :param loss:
            CE loss in batch of training set.
        :param accuracy:
            accuracy in batch of training set.
        :param log:
            indicate if values are printed for user.
        """

        self.step_logs['loss'].append(loss)
        self.step_logs['accuracy'].append(accuracy)

        if log:
            sys.stdout.write(f'\r[{partial}/{total}] ({it_per_sec:.2f}it/s) loss: {loss:.4f} | accu: {accuracy:.4f}')

    def log_model(self,
                  epoch: int,
                  train_time: float,
                  best: bool,
                  model: nn.Module):
        """
        Save model to either best or last register.

        :param epoch:
            epoch where model to be saved was obtained.
        :param train_time:
            training time required to obtain this model.
        :param best:
            indicate if model is saved on best or last register.
        :param model:
            model to be saved.
        """

        # select register where to save model
        reg_key = 'best' if best else 'last'
        self.models[reg_key] = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': self.epoch_logs['train_loss'][-1],
            'val_loss': self.epoch_logs['val_loss'][-1],
            'train_acc': self.epoch_logs['train_accuracy'][-1],
            'val_acc': self.epoch_logs['val_accuracy'][-1],
            'train_time': train_time
        }

    def log_hyperparams(self, hyperparams: dict, log: bool = True):
        """
        Register hyper parameters used for training a model.

        :param hyperparams:
            dict containing hyper parameters as (key, value) pairs.
        :param log:
            indicate if values are printed for user.
        """

        self.hyperparameters.update(hyperparams)
        if log:
            print(dedent(f"""
            Model - {self.hyperparameters['model']} - {self.hyperparameters['n_params']} parameters
            
            Epochs (Max) - {self.hyperparameters['epochs']}
            Batch size - {self.hyperparameters['batch_size']}
            Initial lr -  {self.hyperparameters['lr']}
            Early stopping - {self.hyperparameters['es']} patience epochs 
            
            Images range per class:
                Train - {self.hyperparameters['train_range']}
                Validation - {self.hyperparameters['val_range']}
            """))

    def save_logs(self, base_path: str or Path):
        """
        Save training logs, models and hyper-parameters.

        :param base_path:
            folder path where to store training logs.
        """

        # main paths
        metrics_path = base_path / 'metrics'
        models_path = base_path / 'models'

        # construct dirs if they don't exist
        metrics_path.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)

        # save training logs and hyper-parameters
        torch.save(obj=self.epoch_logs, f=metrics_path / 'epoch_logs.pt')
        torch.save(obj=self.step_logs, f=metrics_path / 'step_logs.pt')
        torch.save(obj=self.hyperparameters, f=models_path / 'hyperparams.pt')

        # save both best and last model
        torch.save(obj=self.models['best'], f=models_path / 'best.pt')
        torch.save(obj=self.models['last'], f=models_path / 'last.pt')
