import torch
import typing
import numpy as np
from tqdm import tqdm

from .metrics import Metric
from .callbacks import Callback
from .dataProvider import DataProvider

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


class MetricsHandler:
    def __init__(self, metrics: typing.List[Metric]):
        self.metrics = metrics

        # Validate metrics
        if not all(isinstance(m, Metric) for m in self.metrics):
            raise TypeError("all items in the metrics argument must be of type Metric (Check mltu.metrics.metrics.py for more information)")
        
        self.train_results_dict = {'loss': None}
        self.train_results_dict.update({metric.name: None for metric in self.metrics})
        
        self.val_results_dict = {'val_loss': None}
        self.val_results_dict.update({"val_" + metric.name: None for metric in self.metrics})

    def update(self, target, output):
        for metric in self.metrics:
            metric.update(output, target)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def results(self, loss, train: bool=True):
        if train:
            self.train_results_dict['loss'] = loss
            for metric in self.metrics:
                self.train_results_dict[metric.name] = metric.result()
            return self.train_results_dict
        
        self.val_results_dict['val_loss'] = loss
        for metric in self.metrics:
            self.val_results_dict["val_" + metric.name] = metric.result()  
        return self.val_results_dict
    
    def description(self, epoch: int=None, train: bool=True):
        epoch_desc = f"Epoch {epoch} - " if epoch is not None else "          "
        dict = self.train_results_dict if train else self.val_results_dict
        
        return epoch_desc + " - ".join([f"{k}: {v:.4f}" for k, v in dict.items()])
    

class Model:
    def __init__(
            self, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            loss: typing.Callable,
            metrics: typing.List[Metric] = [],
            callbacks: typing.List[Callback] = []
        ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.metrics = MetricsHandler(metrics)
        self.callbacks = callbacks

        # get device on which model is running
        self._device = next(self.model.parameters()).device

        self.validate()

    def validate(self):
        """ Validate model, optimizer, loss, metrics and callbacks"""
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model argument must be a torch.nn.Module")
        
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer argument must be a torch.optim.Optimizer")
            
        if not isinstance(self.callbacks, list):
            raise TypeError("callbacks argument must be a list")

        # Validate callbacks
        if not all(isinstance(c, Callback) for c in self.callbacks):
            raise TypeError("all items in the callbacks argument must be of type Callback (Check mltu.callbacks.callbacks.py for more information)")

    @check_data_type
    @check_data_device
    def train_step(self, data, target):
        # TODO before training step, call callbacks
        self.optimizer.zero_grad()
        output = self.model(data)
        # TODO after training step, call callbacks
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

        self.metrics.update(target, output)

        return loss.item()
    
    @check_data_type
    @check_data_device
    def test_step(self, data, target):
        # TODO before testing step, call callbacks
        output = self.model(data)
        # TODO after testing step, call callbacks
        loss = self.loss(output, target)

        self.metrics.update(target, output)

        return loss.item()
    
    def train(self, dataProvider: DataProvider):
        self.model.train()
        losses = []
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar):
            loss = self.train_step(data, target)
            losses.append(loss)

            # get training results of one step
            results_dict = self.metrics.results(np.mean(losses), train=True)
            description = self.metrics.description(epoch=self._epoch, train=True)

            # update progress bar description
            pbar.set_description(description)

        # reset metrics after each training epoch
        self.metrics.reset()

        return results_dict

    def test(self, dataProvider: DataProvider):
        self.model.eval()
        losses = []
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar):
            loss = self.test_step(data, target)
            losses.append(loss)

            # get testing results of one step
            results_dict = self.metrics.results(np.mean(losses), train=False)
            description = self.metrics.description(train=False)

            # update progress bar description
            pbar.set_description(description)

        # reset metrics after each test epoch
        self.metrics.reset()

        return results_dict
    
    def fit(self, train_dataProvider: DataProvider, test_dataProvider: DataProvider, epochs: int, initial_epoch:int = 1):
        self._epoch = initial_epoch
        for epoch in range(initial_epoch, initial_epoch + epochs):
            train_logs = self.train(train_dataProvider)
            val_logs = self.test(test_dataProvider)
            self._epoch += 1