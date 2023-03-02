import torch
import typing
import numpy as np
from tqdm import tqdm

from .metrics import Metric
from .callbacks import Callback
from .dataProvider import DataProvider
from .handlers import MetricsHandler, CallbacksHandler

def check_data_type(func):
    """ Check if data is of type torch.Tensor, if not convert it to torch.Tensor"""
    def wrapper(self, data, target):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target)

        if data.dtype != torch.float32:
            data = data.float()

        return func(self, data, target)
    return wrapper

def check_data_device(func):
    """ Check if data is on the same device as model, if not move it to the device"""
    def wrapper(self, data, target):
        if data.device != self._device:
            data = data.to(self._device)

        if target.device != self._device:
            target = target.to(self._device)

        return func(self, data, target)
    return wrapper


class Model:
    def __init__(
            self, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            loss: typing.Callable,
            metrics: typing.List[Metric] = [],
        ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.metrics = MetricsHandler(metrics)

        self.stop_training = False
        # get device on which model is running
        self._device = next(self.model.parameters()).device

        self.validate()

    def validate(self):
        """ Validate model, optimizer, loss, metrics and callbacks"""
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model argument must be a torch.nn.Module")
        
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer argument must be a torch.optim.Optimizer")

    @check_data_type
    @check_data_device
    def train_step(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

        self.metrics.update(target, output)

        return loss
    
    @check_data_type
    @check_data_device
    def test_step(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        self.metrics.update(target, output)

        return loss
    
    def train(self, dataProvider: DataProvider):
        self.model.train()
        losses = []
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar):
            self.callbacks.on_batch_begin(step)
            self.callbacks.on_train_batch_begin(step)

            loss = self.train_step(data, target)
            losses.append(loss.item())

            # get training results of one step
            logs = self.metrics.results(np.mean(losses), train=True)
            description = self.metrics.description(epoch=self._epoch, train=True)

            # update progress bar description
            pbar.set_description(description)

            self.callbacks.on_train_batch_end(step, logs=logs)
            self.callbacks.on_batch_end(step, logs=logs)

        # reset metrics after each training epoch
        self.metrics.reset()

        return logs

    def test(self, dataProvider: DataProvider):
        self.model.eval()
        losses = []
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar):
            self.callbacks.on_batch_begin(step)
            self.callbacks.on_test_batch_begin(step)

            loss = self.test_step(data, target)
            losses.append(loss.item())

            # get testing results of one step
            logs = self.metrics.results(np.mean(losses), train=False)
            description = self.metrics.description(train=False)

            # update progress bar description
            pbar.set_description(description)

            self.callbacks.on_train_batch_end(step, logs=logs)
            self.callbacks.on_batch_end(step, logs=logs)

        # reset metrics after each test epoch
        self.metrics.reset()

        return logs
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def fit(self, train_dataProvider: DataProvider, test_dataProvider: DataProvider, epochs: int, initial_epoch:int = 1, callbacks: typing.List[Callback] = []):
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