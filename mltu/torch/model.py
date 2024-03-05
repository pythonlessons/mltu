import os
import math
import torch
import typing
import traceback
import numpy as np
from torch import nn
from copy import deepcopy
from qqdm import qqdm, format_str
from pathlib import Path

from .metrics import Metric
from .callbacks import Callback
from .dataProvider import DataProvider
from .handlers import MetricsHandler, CallbacksHandler

def toTorch(data: np.ndarray, target: np.ndarray) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """ Check if data is of type torch.Tensor, if not convert it to torch.Tensor

    Args:
        data (np.ndarray): data to be converted
        target (np.ndarray): target to be converted

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: converted data and target
    """
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)

    if not isinstance(target, (torch.Tensor, dict)):
        target = torch.from_numpy(target)

    if data.dtype != torch.float32:
        data = data.float()

    return data, target


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


class Model:
    """ Model class for training and testing PyTorch neural networks"""
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        loss: typing.Callable,
        metrics: typing.List[Metric] = [],
        mixed_precision: bool = False,
        scaler = None,
        log_errors: bool = True,
        output_path: str = None,
        clip_grad_norm: float = None,
        ema = False,
        ):
        """ Initialize model class

        Attributes:
            model (torch.nn.Module): PyTorch neural network
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            loss (typing.Callable): loss function
            metrics (typing.List[Metric], optional): list of metrics. Defaults to [].
            mixed_precision (bool, optional): whether to use mixed precision. Defaults to False.
            scaler (torch.cuda.amp.GradScaler, optional): PyTorch GradScaler. Defaults to None.
            log_errors (bool, optional): whether to log errors. Defaults to True.
            output_path (str, optional): path to save model. Defaults to None.
            clip_grad_norm (float, optional): gradient clipping value. Defaults to None.
            ema (bool, optional): whether to use Exponential Moving Average. Defaults to False.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        # get device on which model is running
        self._device = next(self.model.parameters()).device

        self.metrics = MetricsHandler(metrics)

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else scaler
        self.log_errors = log_errors

        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            if self.output_path.suffix == "":
                self.output_path = Path(os.path.join(self.output_path, "model.pt"))
            os.makedirs(self.output_path.parent, exist_ok=True)

        self.stop_training = False
        self.clip_grad_norm = clip_grad_norm
        self.ema = ModelEMA(self.model) if ema else None

        self.validate()

    @property
    def device(self):
        return self._device

    def validate(self):
        """ Validate model, optimizer"""
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model argument must be a torch.nn.Module")
        
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer argument must be a torch.optim.Optimizer")
        
    def toDevice(self, data: np.ndarray, target: np.ndarray) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """ Check if data is on the same device as model, if not move it to the device

        Args:
            data (np.ndarray): data to be moved
            target (np.ndarray): target to be moved

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor]: moved data and target
        """
        try:
            if data.device != self._device:
                data = data.to(self._device)

            if target.device != self._device:
                target = target.to(self._device)
        except AttributeError:
            if self.log_errors:
                print("Data and target must be of type torch.Tensor")

        return data, target

    def train_step(
        self, 
        data: typing.Union[np.ndarray, torch.Tensor], 
        target: typing.Union[np.ndarray, torch.Tensor]
        ) -> torch.Tensor:
        """ Perform one training step

        Args:
            data (typing.Union[np.ndarray, torch.Tensor]): training data
            target (typing.Union[np.ndarray, torch.Tensor]): training target

        Returns:
            torch.Tensor: loss
        """
        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(loss, tuple):
                    loss, loss_info = loss[0], loss[1:]
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        else:
            output = self.model(data)
            loss = self.loss(output, target)
            if isinstance(loss, tuple):
                loss, loss_info = loss[0], loss[1:]
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()

        if self.ema:
            try:
                self.ema.update(self.model)
            except RuntimeError:
                self.ema = ModelEMA(self.model)
                self.ema.update(self.model)

        if self._device.type == "cuda":
            torch.cuda.synchronize() # synchronize after each forward and backward pass

        self.metrics.update(target, output, model=self.model, loss_info=loss_info)

        return loss
    
    def test_step(
        self, 
        data: typing.Union[np.ndarray, torch.Tensor], 
        target: typing.Union[np.ndarray, torch.Tensor]
        ) -> torch.Tensor:
        """ Perform one validation step

        Args:
            data (typing.Union[np.ndarray, torch.Tensor]): validation data
            target (typing.Union[np.ndarray, torch.Tensor]): validation target

        Returns:
            torch.Tensor: loss
        """
        output = self.ema.ema(data) if self.ema else self.model(data)
        loss = self.loss(output, target)
        if isinstance(loss, tuple):
            loss, loss_info = loss[0], loss[1:]

        self.metrics.update(target, output, model=self.model, loss_info=loss_info)

        # clear GPU memory cache after each validation step
        torch.cuda.empty_cache()

        return loss
    
    def train(self, dataProvider: DataProvider):
        """ Perform one training epoch
        
        Args:
            dataProvider (DataProvider): data provider for training data

        Returns:
            dict: training results
        """
        # set model to training mode
        self.model.train()

        loss_sum = 0
        pbar = qqdm(dataProvider, total=len(dataProvider), desc=format_str('bold', f"Epoch {self._epoch}: "))
        for step, (data, target) in enumerate(pbar, start=1):
            self.callbacks.on_batch_begin(step, logs=None, train=True)

            data, target = self.toDevice(*toTorch(data, target))
            loss = self.train_step(data, target)
            loss_sum += loss.item()
            loss_mean = loss_sum / step

            # get training results of one step
            logs = self.metrics.results(loss_mean, train=True)

            # log learning rate into logs
            if len(self.optimizer.param_groups) > 1:
                lr_logs = {f"lr{i}": round(group["lr"], 6) for i, group in enumerate(self.optimizer.param_groups)}
                logs.update(lr_logs)
            else:
                logs["lr"] = round(self.optimizer.param_groups[0]["lr"], 6)

            # update progress bar description
            pbar.set_description(desc=format_str('bold', f"Epoch {self._epoch}: "))
            pbar.set_infos(logs)

            self.callbacks.on_batch_end(step, logs=logs, train=True)

        # reset metrics after each training epoch
        self.metrics.reset()

        # call on_epoch_end of data provider
        dataProvider.on_epoch_end()

        return logs

    def test(self, dataProvider: DataProvider):
        """ Perform one validation epoch

        Args:
            dataProvider (DataProvider): data provider for validation data

        Returns:
            dict: validation results
        """
        # set model to evaluation mode
        self.model.eval()
        loss_sum = 0
        pbar = qqdm(dataProvider, total=len(dataProvider), desc=format_str('bold', 'Description'))
        # disable autograd and gradient computation in PyTorch
        with torch.no_grad():
            for step, (data, target) in enumerate(pbar, start=1):
                self.callbacks.on_batch_begin(step, logs=None, train=False)

                data, target = self.toDevice(*toTorch(data, target))
                loss = self.test_step(data, target)
                loss_sum += loss.item()
                loss_mean = loss_sum / step

                # get testing results of one step
                logs = self.metrics.results(loss_mean, train=False)

                # update progress bar description
                pbar.set_description(f"Epoch {self._epoch}: ")
                pbar.set_infos(logs)

                self.callbacks.on_batch_end(step, logs=logs, train=False)

        # reset metrics after each test epoch
        self.metrics.reset()

        # call on_epoch_end of data provider
        dataProvider.on_epoch_end()

        return logs
    
    def save(self, path: str=None):
        """ Save model state dict to file

        Args:
            path (str): path to file
        """
        if not path and not self.output_path:
            print("Path to file is not provided, model will not be saved") # replace to error logging
            return
        
        model_to_save = self.ema.ema if self.ema else self.model
        
        output_path = Path(path or self.output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        model_to_save.eval()
        try:
            torch.save(model_to_save.state_dict(), output_path)
            return str(output_path)
        except Exception:
            traceback.print_exc()
            torch.save(model_to_save, output_path.with_suffix(".pth"))
            return str(output_path.with_suffix(".pth"))
    
    def fit(
        self, 
        train_dataProvider: DataProvider, 
        test_dataProvider: DataProvider=None, 
        epochs: int=10, 
        initial_epoch: int = 1, 
        callbacks: typing.List[Callback] = []
        ) -> dict:
        """ Train model for a given number of epochs
        
        Args:
            train_dataProvider (DataProvider): data provider for training data
            test_dataProvider (DataProvider): data provider for validation data
            epochs (int): number of epochs
            initial_epoch (int, optional): initial epoch. Defaults to 1.
            callbacks (typing.List[Callback], optional): list of callbacks. Defaults to [].

        Returns:
            dict: training results
        """
        self._epoch = initial_epoch
        history = {}
        self.callbacks = CallbacksHandler(self, callbacks)
        self.callbacks.on_train_begin()
        for epoch in range(initial_epoch, initial_epoch + epochs):
            self.callbacks.on_epoch_begin(epoch)

            train_logs = self.train(train_dataProvider)
            val_logs = self.test(test_dataProvider) if test_dataProvider else {}

            logs = {**train_logs, **val_logs}
            self.callbacks.on_epoch_end(epoch, logs=logs)

            if self.stop_training:
                break

            history[epoch] = logs
            self._epoch += 1

        self.callbacks.on_train_end(logs)

        return history
    
    def evaluate(
        self, 
        dataProvider: DataProvider, 
        initial_epoch: int = 1,
        callbacks: typing.List[Callback] = [],
        ) -> dict:
        """ Evaluate model on validation data

        Args:
            dataProvider (DataProvider): data provider for validation data

        Returns:
            dict: validation results
        """
        self._epoch = initial_epoch
        self.callbacks = CallbacksHandler(self, callbacks)
        self.model.eval()
        logs = self.test(dataProvider)
        return logs