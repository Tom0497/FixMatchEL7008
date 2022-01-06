import sys
from textwrap import dedent

from train.summary.summary import Summary


class SummarySSL(Summary):
    """
    Logger and register for model trainer.

    Subclass of Summary, this class adds metrics and hyper parameters
    practical for a semi-supervised learning scheme, in specific, the
    FixMatch algorithm.

    Loss now has a supervised and an unsupervised component, SSL specific
    metrics such as impurity and mask rate are also considered.

    FixMatch itself adds three hyper parameters to the training scheme,
    tau, mu and lambda u.
    """

    def __init__(self):
        """
        Constructor of SummarySSL.
        """

        # first construct parent class
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
        """
        Log and display important metrics from a step or batch.

        :param partial:
            number of images already used from training set in current epoch.
        :param it_per_sec:
            iterations per second estimation.
        :param total:
            number of images in training set.
        :param loss:
            loss function value in batch of training set.
        :param accuracy:
            accuracy in batch of training set.
        :param log:
            indicate if values are printed for user.

        :param su_loss:
            supervised component of loss function.
        :param un_loss:
            unsupervised component of loss function
        :param impurity:
            error rate of unlabeled data that falls above threshold.
        :param mask_rate:
            proportion of unlabeled examples that are masked out.
        """

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
                f'loss: {loss:.4f} = {su_loss:.4f}+{lam}*{un_loss:.4f} '
                f'| accu: {accuracy:.4f} '
                f'| im: {impurity:.4f} '
                f'| mr: {mask_rate:.4f} ')

    def log_epoch_ssl(self,
                      train_su_loss: float,
                      train_un_loss: float,
                      train_impurity: float,
                      train_mask_rate: float):
        """
        Log metrics specific for fixmatch training from an epoch.

        :param train_su_loss:
            supervised component of loss function.
        :param train_un_loss:
            unsupervised component of loss function.
        :param train_impurity:
            error rate of unlabeled data that falls above threshold.
        :param train_mask_rate:
            proportion of unlabeled examples that are masked out.
        """

        self.epoch_logs['train_su_loss'].append(train_su_loss)
        self.epoch_logs['train_un_loss'].append(train_un_loss)
        self.epoch_logs['train_impurity'].append(train_impurity)
        self.epoch_logs['train_mask_rate'].append(train_mask_rate)

    def log_hyperparams(self, hyperparams: dict, log: bool = True):
        """
        Register hyper parameters used for training a model.

        :param hyperparams:
            dict containing hyper parameters as (key, value) pairs.
        :param log:
            indicate if values are printed for user.
        """

        # parent method carries the logging process
        super().log_hyperparams(hyperparams=hyperparams, log=False)

        if log:
            # printing differs given fixmatch consider more hyper parameters
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
