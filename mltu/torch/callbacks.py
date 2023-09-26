import os
import onnx
import logging
import numpy as np
from datetime import datetime

import torch.onnx
from torch.utils.tensorboard import SummaryWriter

class Callback:
    """ Base class used to build new callbacks."""
    def __init__(
        self, 
        monitor: str = "val_loss"
    ) -> None:
        self.monitor = monitor
        logging.basicConfig()
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
        mode: str = "min",
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

def assign_mode(mode: str):
    if mode not in ["min", "max", "max_equal", "min_equal"]:
        raise ValueError(
            "ModelCheckpoint mode %s is unknown, "
            "please choose one of min, max, max_equal, min_equal" % mode
        )

    if mode == "min": return np.less
    elif mode == "max": return np.greater
    elif mode == "min_equal": return np.less_equal
    elif mode == "max_equal": return np.greater_equal

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

        self.monitor_op = assign_mode(self.mode)
        
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


class TensorBoard(Callback):
    """ TensorBoard basic visualizations. """
    def __init__(
            self, 
            log_dir: str = "logs", 
            comment: str = None, 
            histogram: bool=False,
            train_name: str = "train",
            val_name: str = "test",
            train_writer: SummaryWriter = None,
            val_writer: SummaryWriter = None,
        ):
        """ TensorBoard basic visualizations.
        
        Args:
            log_dir (str, optional): the path of the directory where to save the log files to be parsed by TensorBoard. Defaults to "logs".
            comment (str, optional): comment to append to the default log_dir. Defaults to None.
            histogram (bool, optional): if True, histogram of the model's parameters will be saved. Defaults to False.
            train_name (str, optional): name of the training writer. Defaults to "train".
            val_name (str, optional): name of the validation writer. Defaults to "test".
            train_writer (SummaryWriter, optional): training writer. Defaults to None.
            val_writer (SummaryWriter, optional): validation writer. Defaults to None.
        """
        super(TensorBoard, self).__init__()

        self.log_dir = log_dir

        self.train_writer = train_writer
        self.val_writer = val_writer
        self.comment = str(comment) if not None else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.histogram = histogram

        self.train_name = train_name
        self.val_name = val_name

    def on_train_begin(self, logs=None):
        if self.train_writer is None:
            train_dir = self.log_dir + "/" + self.train_name
            os.makedirs(train_dir, exist_ok=True)
            self.train_writer = SummaryWriter(train_dir, comment=self.comment)

        if self.val_writer is None:
            val_dir = self.log_dir + "/" + self.val_name
            os.makedirs(val_dir, exist_ok=True)
            self.val_writer = SummaryWriter(val_dir, comment=self.comment)

    def update_lr(self, epoch: int):
        for param_group in self.model.optimizer.param_groups:
            if self.train_writer:
                self.train_writer.add_scalar("learning_rate", param_group["lr"], epoch)

    # def update_histogram(self, epoch: int):
    #     for name, param in self.model.model.named_parameters():
    #         self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def parse_key(self, key: str):
        if key.startswith("val_"):
            return self.val_name, key[4:]
        else:
            return self.train_name, key

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            _type, key = self.parse_key(key)
            if _type == self.train_name:
                self.train_writer.add_scalar(key, value, epoch)
            else:
                self.val_writer.add_scalar(key, value, epoch)

        self.update_lr(epoch)
        if self.histogram:
            pass
            # TODO
            # self.update_histogram(epoch)

    def on_train_end(self, logs=None):
        self.train_writer.close()
        self.val_writer.close()


class Model2onnx(Callback):
    """Converts the model from PyTorch to ONNX format after training."""

    def __init__(
            self,
            saved_model_path: str,
            input_shape: tuple,
            export_params: bool = True,
            opset_version: int = 14,
            do_constant_folding: bool = True,
            input_names: list = ["input"],
            output_names: list = ["output"],
            dynamic_axes: dict = {"input": {0: "batch_size"},
                                  "output": {0: "batch_size"}},
            verbose: bool = False,
            metadata: dict = None,
    ) -> None:
        """ Converts the model from PyTorch to ONNX format after training.

        Args:
            saved_model_path (str): path to the saved model
            input_shape (tuple): input shape of the model
            export_params (bool, optional): if True, all model parameters will be exported. Defaults to True.
            opset_version (int, optional): the ONNX version to export the model to. Defaults to 14.
            do_constant_folding (bool, optional): whether to execute constant folding for optimization. Defaults to True.
            input_names (list, optional): the model's input names. Defaults to ['input'].
            output_names (list, optional): the model's output names. Defaults to ['output'].
            dynamic_axes (dict, optional): dictionary specifying dynamic axes. Defaults to {'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}.
            verbose (bool, optional): if True, information about the conversion will be printed. Defaults to False.
            metadata (dict, optional): dictionary containing model metadata. Defaults to None.
        """
        super().__init__()
        self.saved_model_path = saved_model_path
        self.input_shape = input_shape
        self.export_params = export_params
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.verbose = verbose
        self.metadata = metadata
        
        self.onnx_model_path = saved_model_path.replace(".pt", ".onnx")

    def on_train_end(self, logs=None):
        self.model.model.load_state_dict(torch.load(self.saved_model_path))

        # place model on cpu
        self.model.model.to("cpu")

        # set the model to inference mode
        self.model.model.eval()
        
        # convert the model to ONNX format
        dummy_input = torch.randn(self.input_shape)

        # Export the model
        torch.onnx.export(
            self.model.model,               
            dummy_input,                         
            self.onnx_model_path,   
            export_params=self.export_params,        
            opset_version=self.opset_version,          
            do_constant_folding=self.do_constant_folding,  
            input_names = self.input_names,   
            output_names = self.output_names, 
            dynamic_axes = self.dynamic_axes,
            )
        
        if self.verbose:
            self.logger.info(f"Model saved to {self.onnx_model_path}")

        if self.metadata and isinstance(self.metadata, dict):
            # Load the ONNX model
            onnx_model = onnx.load(self.onnx_model_path)

            # Add the metadata dictionary to the model's metadata_props attribute
            for key, value in self.metadata.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)

            # Save the modified ONNX model
            onnx.save(onnx_model, self.onnx_model_path)

class ReduceLROnPlateau(Callback):
    """ Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs,
    the learning rate is reduced.
    """
    def __init__(
        self, 
        monitor: str = "val_loss", 
        factor: float = 0.1, 
        patience: int = 10, 
        min_lr: float = 1e-6, 
        mode: str = "min",
        verbose: int = False,
        ) -> None:
        """ Reduce learning rate when a metric has stopped improving.
        
        Args:
            monitor (str, optional): quantity to be monitored. Defaults to "val_loss".
            factor (float, optional): factor by which the learning rate will be reduced. Defaults to 0.1.
            patience (int, optional): number of epochs with no improvement after which learning rate will be reduced. Defaults to 10.
            min_lr (float, optional): lower bound on the learning rate. Defaults to 1e-6.
            verbose (int, optional): verbosity mode. Defaults to 0.
            mode (str, optional): one of {min, max, max_equal, min_equal}. Defaults to "min". 
        """
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.mode = mode

        self.monitor_op = assign_mode(self.mode)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.inf if self.mode == "min" or self.mode == "min_equal" else -np.Inf

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                current_lr = self.model.optimizer.param_groups[0]["lr"]
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.param_groups[0]["lr"] = new_lr
                if self.verbose:
                    self.logger.info(f"Epoch {epoch}: reducing learning rate to {new_lr}.")


class WarmupCosineDecay(Callback):
    """ Cosine decay learning rate scheduler with warmup

    Args:
        lr_after_warmup (float): Learning rate after warmup
        final_lr (float): Final learning rate
        warmup_epochs (int): Number of warmup epochs
        decay_epochs (int): Number of decay epochs
        initial_lr (float, optional): Initial learning rate. Defaults to 0.0.
        verbose (bool, optional): Whether to print learning rate. Defaults to False.
        warmup_steps (int, optional): Number of warmup steps. Defaults to None.
    """
    def __init__(
            self, 
            lr_after_warmup: float, 
            final_lr: float, 
            warmup_epochs: int, 
            decay_epochs: int, 
            initial_lr: float=0.0, 
            warmup_steps: int=None,
            verbose=False
        ) -> None:
        super(WarmupCosineDecay, self).__init__()
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.step = None

        self.warmup_lrs = np.linspace(self.initial_lr, self.lr_after_warmup, self.warmup_epochs)
        if warmup_steps:
            self.step = 0
            self.warmup_epochs = 0
            self.warmup_lrs = np.linspace(self.initial_lr, self.lr_after_warmup, warmup_steps)

    def on_epoch_begin(self, epoch: int, logs: dict=None):
        """ Adjust learning rate at the beginning of each epoch """

        if self.warmup_steps:
            return logs

        if epoch >= self.warmup_epochs + self.decay_epochs:
            return logs

        if epoch <= self.warmup_epochs:
            lr = self.warmup_lrs[epoch-1]
        else:
            progress = (epoch - self.warmup_epochs) / self.decay_epochs
            lr = self.final_lr + 0.5 * (self.lr_after_warmup - self.final_lr) * (1 + np.cos(np.pi * progress))

        self.model.optimizer.param_groups[0]["lr"] = lr
        
        if self.verbose:
            self.logger.info(f"Epoch {epoch} - Learning Rate: {lr}")

    def on_train_batch_begin(self, batch: int, logs: dict=None):
        if self.warmup_steps and self.step is not None:
            if self.step < self.warmup_steps:
                self.model.optimizer.param_groups[0]["lr"] = self.warmup_lrs[self.step]
                self.step += 1
            else:
                self.step = None
    
    def on_epoch_end(self, epoch: int, logs: dict=None):
        logs = logs or {}
        
        # Log the learning rate value
        logs["lr"] = self.model.optimizer.param_groups[0]["lr"]
        
        return logs