import sys
from textwrap import dedent

from train.summary.summary import Summary


class SummarySSL(Summary):

    def __init__(self):
        super().__init__()

        # new training logs for ssl
        self.epoch_logs.update(dict(
            train_su_loss=[],
            train_un_loss=[],
            train_impurity=[],
            train_mask_rate=[]
        ))
        self.step_logs.update(dict(
            impurity=[],
            mask_rate=[],
            su_loss=[],
            un_loss=[]
        ))

        # training hyper parameters
        self.hyperparameters.update(dict(
            unlabeled_range=None,
            mu=None,
            tau=None,
            lambda_u=None
        ))

    def log_step_ssl(self,
                     partial: int,
                     total: int,
                     loss: float,
                     accuracy: float,
                     su_loss: float,
                     un_loss: float,
                     impurity: float,
                     mask_rate: float,
                     it_per_sec: float,
                     log: bool = True):

        super().log_step(partial=partial,
                         total=total,
                         loss=loss,
                         accuracy=accuracy,
                         it_per_sec=it_per_sec,
                         log=False)

        self.step_logs['su_loss'].append(su_loss)
        self.step_logs['un_loss'].append(un_loss)
        self.step_logs['impurity'].append(impurity)
        self.step_logs['mask_rate'].append(mask_rate)

        if log:
            lam = self.hyperparameters['lambda_u']
            sys.stdout.write(
                f'\r[{partial}/{total}] ({it_per_sec:.2f}it/s) '
                f'loss: {loss:.4f} = {su_loss:.4f} + {lam}*{un_loss:.4f}'
                f'| accu: {accuracy:.4f}')

    def log_epoch_ssl(self,
                      train_su_loss: float,
                      train_un_loss: float,
                      train_impurity: float,
                      train_mask_rate: float):

        self.epoch_logs['train_su_loss'].append(train_su_loss)
        self.epoch_logs['train_un_loss'].append(train_un_loss)
        self.epoch_logs['train_impurity'].append(train_impurity)
        self.epoch_logs['train_mask_rate'].append(train_mask_rate)

    def log_hyperparams(self, hyperparams: dict, log: bool = True):

        super().log_hyperparams(hyperparams=hyperparams, log=False)

        if log:
            print(dedent(f"""
            Model - {self.hyperparameters['model']} - {self.hyperparameters['n_params']} parameters

            Epochs (Max) - {self.hyperparameters['epochs']}
            Batch size - {self.hyperparameters['batch_size']}
            Initial lr -  {self.hyperparameters['lr']}
            Early stopping - {self.hyperparameters['es']} patience epochs 
            
            FixMatch Hyper-params
                lambda_u - {self.hyperparameters['lambda_u']}
                tau - {self.hyperparameters['tau']}
                mu - {self.hyperparameters['mu']}

            Images range per class:
                Labeled - {self.hyperparameters['train_range']}
                Unlabeled - {self.hyperparameters['unlabeled_range']}
                Validation - {self.hyperparameters['val_range']}
            """))
