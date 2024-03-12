import typing

from .metrics import Metric
from .callbacks import Callback

class MetricsHandler:
    """ Metrics handler class for training and testing loops"""
    def __init__(self, metrics: typing.List[Metric]):
        self.metrics = metrics

        # Validate metrics
        if not all(isinstance(m, Metric) for m in self.metrics):
            raise TypeError("all items in the metrics argument must be of type Metric (Check mltu.metrics.metrics.py for more information)")
        
        self.train_results_dict = {"loss": None}
        self.train_results_dict.update({metric.name: None for metric in self.metrics})
        
        self.val_results_dict = {"val_loss": None}
        self.val_results_dict.update({"val_" + metric.name: None for metric in self.metrics})

    def update(self, target, output, **kwargs):
        for metric in self.metrics:
            metric.update(output, target, **kwargs)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def results(self, loss, train: bool=True):
        suffix = "val_" if not train else ""
        results_dict = self.val_results_dict if not train else self.train_results_dict
        results_dict[suffix + "loss"] = loss
        for metric in self.metrics:
            result = metric.result()
            if result:
                if isinstance(result, dict):
                    for k, v in result.items():
                        results_dict[suffix + k] = v
                else:
                    results_dict[suffix + metric.name] = result

        logs = {k: round(v, 4) for k, v in results_dict.items() if v is not None}
        return logs
    
    def description(self, epoch: int=None, train: bool=True):
        epoch_desc = f"Epoch {epoch} - " if epoch is not None else "          "
        dict = self.train_results_dict if train else self.val_results_dict
        return epoch_desc + " - ".join([f"{k}: {v:.4f}" for k, v in dict.items() if v])
    

class CallbacksHandler:
    """ Callbacks handler class for training and testing loops"""
    def __init__(self, model, callbacks: typing.List[Callback]):
        self.callbacks = callbacks

        # Validate callbacks
        if not all(isinstance(c, Callback) for c in self.callbacks):
            raise TypeError("all items in the callbacks argument must be of type Callback (Check mltu.torch.callbacks.py for more information)")
        
        for callback in self.callbacks:
            callback.model = model
        
    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_test_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_batch_begin(self, batch: int, logs=None, train: bool=True):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

            if train:
                callback.on_train_batch_begin(batch, logs)
            else:
                callback.on_test_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs=None, train: bool=True):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

            if train:
                callback.on_train_batch_end(batch, logs)
            else:
                callback.on_test_batch_end(batch, logs)