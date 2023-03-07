import os
import logging
import numpy as np

class Callback:
    """ Base class used to build new callbacks."""
    def __init__(
        self, 
        monitor: str = "val_loss"
    ) -> None:
        self.monitor = monitor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch: int, logs=None):
        pass

    def on_train_batch_end(self, batch: int, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch: int, logs=None):
        pass

    def on_test_batch_end(self, batch: int, logs=None):
        pass

    def on_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_epoch_end(self, epoch: int, logs=None):
        pass

    def on_batch_begin(self, batch: int, logs=None):
        pass

    def on_batch_end(self, batch: int, logs=None):
        pass

    def get_monitor_value(self, logs: dict):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

class EarlyStopping(Callback):
    def __init__(
        self, 
        monitor: str = "val_loss",
        min_delta: float = 0.0, 
        patience: int = 0, 
        verbose: bool = False,
        mode: str = "max_equal",
        ):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.wait = None
        self.stopped_epoch = None
        self.best = None

        if self.mode not in ["min", "max", "max_equal", "min_equal"]:
            raise ValueError(
                "EarlyStopping mode %s is unknown, "
                "please choose one of min, max, max_equal, min_equal" % self.mode
            )
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.mode == "min" or self.mode == "min_equal" else -np.Inf

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.mode == "min" and np.less(current, self.best - self.min_delta):
            self.best = current
            self.wait = 0
        elif self.mode == "max" and np.greater(current, self.best + self.min_delta):
            self.best = current
            self.wait = 0
        elif self.mode == "min_equal" and np.less_equal(current, self.best - self.min_delta):
            self.best = current
            self.wait = 0
        elif self.mode == "max_equal" and np.greater_equal(current, self.best + self.min_delta):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            self.logger.info(f"Epoch {self.stopped_epoch}: early stopping")


class ModelCheckpoint(Callback):
    """ ModelCheckpoint callback to save the model after every epoch or the best model across all epochs."""
    def __init__(
        self, 
        filepath: str,
        monitor: str = "val_loss",
        verbose: bool = False,
        save_best_only: bool = True,
        mode: str = "min",
        ) -> None:
        """ ModelCheckpoint callback to save the model after every epoch or the best model across all epochs
        
        Args:
            filepath (str): path to save the model file
            monitor (str, optional): metric to monitor. Defaults to "val_loss".
            verbose (bool, optional): verbosity mode. Defaults to False.
            save_best_only (bool, optional): if True, the latest best model according to the quantity monitored will not be overwritten. Defaults to True.
            mode (str, optional): one of {min, max, max_equal, min_equal}. Defaults to "min".
        """
        super(ModelCheckpoint, self).__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None

        if self.mode not in ["min", "max", "max_equal", "min_equal"]:
            raise ValueError(
                "ModelCheckpoint mode %s is unknown, "
                "please choose one of min, max, max_equal, min_equal" % self.mode
            )
        
        if self.mode == "min": self.monitor_op = np.less
        elif self.mode == "max": self.monitor_op = np.greater
        elif self.mode == "min_equal": self.monitor_op = np.less_equal
        elif self.mode == "max_equal": self.monitor_op = np.greater_equal
        
    def on_train_begin(self, logs=None):
        self.best = np.inf if self.mode == "min" or self.mode == "min_equal" else -np.Inf

        # create directory if not exist
        if not os.path.exists(os.path.dirname(self.filepath)):
            os.makedirs(os.path.dirname(self.filepath))

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            previous = self.best
            self.best = current
            self.save_model(epoch, current, previous)
        else:
            if not self.save_best_only:
                self.save_model(epoch, current, previous=None)

    def save_model(self, epoch: int, best: float, previous: float = None):
        """ Save model to filepath
        
        Args:
            epoch (int): current epoch
            best (float): current best value
            previous (float, optional): previous best value. Defaults to None.
        """
        if self.verbose:
            if previous is None:
                self.logger.info(f"Epoch {epoch}: {self.monitor} got {best:.5f}, saving model to {self.filepath}")
            else:
                self.logger.info(f"Epoch {epoch}: {self.monitor} improved from {previous:.5f} to {best:.5f}, saving model to {self.filepath}")

        self.model.save(self.filepath)