import torch
import typing
import numpy as np
from tqdm import tqdm

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

    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    if data.dtype != torch.float32:
        data = data.float()

    return data, target

class Model:
    """ Model class for training and testing PyTorch neural networks"""
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        loss: typing.Callable,
        metrics: typing.List[Metric] = [],
        mixed_precision: bool = False,
        scaler = None
        ):
        """ Initialize model class

        Attributes:
            model (torch.nn.Module): PyTorch neural network
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            loss (typing.Callable): loss function
            metrics (typing.List[Metric], optional): list of metrics. Defaults to [].
            mixed_precision (bool, optional): whether to use mixed precision. Defaults to False.
            scaler (torch.cuda.amp.GradScaler, optional): PyTorch GradScaler. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.metrics = MetricsHandler(metrics)

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else scaler

        self.stop_training = False
        # get device on which model is running
        self._device = next(self.model.parameters()).device

        self.validate()

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
        if data.device != self._device:
            data = data.to(self._device)

        if target.device != self._device:
            target = target.to(self._device)

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
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        else:
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

        torch.cuda.synchronize() # synchronize after each forward and backward pass

        self.metrics.update(target, output)

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
        output = self.model(data)
        loss = self.loss(output, target)

        self.metrics.update(target, output)

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
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar, start=1):
            self.callbacks.on_batch_begin(step, logs=None, train=True)

            data, target = self.toDevice(*toTorch(data, target))
            loss = self.train_step(data, target)
            loss_sum += loss.item()
            loss_mean = loss_sum / step

            # get training results of one step
            logs = self.metrics.results(loss_mean, train=True)
            description = self.metrics.description(epoch=self._epoch, train=True)

            # update progress bar description
            pbar.set_description(description)

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
        pbar = tqdm(dataProvider, total=len(dataProvider))
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
                description = self.metrics.description(train=False)

                # update progress bar description
                pbar.set_description(description)

                self.callbacks.on_batch_end(step, logs=logs, train=False)

        # reset metrics after each test epoch
        self.metrics.reset()

        # call on_epoch_end of data provider
        dataProvider.on_epoch_end()

        return logs
    
    def save(self, path: str):
        """ Save model state dict to file

        Args:
            path (str): path to file
        """
        torch.save(self.model.state_dict(), path)
    
    def fit(
        self, 
        train_dataProvider: DataProvider, 
        test_dataProvider: DataProvider, 
        epochs: int, 
        initial_epoch:int = 1, 
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
        self.callbacks = CallbacksHandler(self, callbacks)
        self.callbacks.on_train_begin()
        for epoch in range(initial_epoch, initial_epoch + epochs):
            self.callbacks.on_epoch_begin(epoch)

            train_logs = self.train(train_dataProvider)
            val_logs = self.test(test_dataProvider)

            logs = {**train_logs, **val_logs}
            self.callbacks.on_epoch_end(epoch, logs=logs)

            if self.stop_training:
                break

            self._epoch += 1

        self.callbacks.on_train_end(logs)

        return logs