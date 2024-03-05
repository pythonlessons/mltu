import torch
import typing
import numpy as np
from itertools import groupby

from mltu.utils.text_utils import get_cer, get_wer


class Metric:
    """ Base class for all metrics"""
    def __init__(self, name: str) -> None:
        """ Initialize metric with name

        Args:
            name (str): name of metric
        """
        self.name = name

    def reset(self):
        """ Reset metric state to initial values and return metric value"""
        self.__init__()

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        """ Update metric state with new data
        
        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target of data
        """
        pass

    def result(self):
        """ Return metric value"""
        pass


class Accuracy(Metric):
    """ Accuracy metric class
    
    Args:
        name (str, optional): name of metric. Defaults to 'accuracy'.
    """
    def __init__(self, name="accuracy") -> None:
        super(Accuracy, self).__init__(name=name)
        self.correct = 0
        self.total = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        """ Update metric state with new data

        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target of data
        """
        _, predicted = torch.max(output.data, 1)
        self.total += target.size(0)
        self.correct += (predicted == target).sum().item()

    def result(self):
        """ Return metric value"""
        return self.correct / self.total


class CERMetric(Metric):
    """A custom PyTorch metric to compute the Character Error Rate (CER).
    
    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.

    # TODO: implement everything in Torch to avoid converting to numpy
    """
    def __init__(
        self, 
        vocabulary: typing.Union[str, list],
        name: str = "CER"
    ) -> None:
        super(CERMetric, self).__init__(name=name)
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
        """ Reset metric state to initial values"""
        self.cer = 0
        self.counter = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        """ Update metric state with new data

        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target of data
        """
        # convert to numpy
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # use argmax to find the index of the highest probability
        argmax_preds = np.argmax(output, axis=-1)
        
        # use groupby to find continuous same indexes
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

        # convert indexes to strings
        output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in target]

        cer = get_cer(output_texts, target_texts)

        self.cer += cer
        self.counter += 1

    def result(self) -> float:
        """ Return metric value"""
        return self.cer / self.counter
    

class WERMetric(Metric):
    """A custom PyTorch metric to compute the Word Error Rate (WER).
    
    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.

    # TODO: implement everything in Torch to avoid converting to numpy
    """
    def __init__(
        self, 
        vocabulary: typing.Union[str, list],
        name: str = "WER"
    ) -> None:
        super(WERMetric, self).__init__(name=name)
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
        """ Reset metric state to initial values"""
        self.wer = 0
        self.counter = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        """ Update metric state with new data

        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target of data
        """
        # convert to numpy
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # use argmax to find the index of the highest probability
        argmax_preds = np.argmax(output, axis=-1)
        
        # use groupby to find continuous same indexes
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

        # convert indexes to strings
        output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in target]

        wer = get_wer(output_texts, target_texts)

        self.wer += wer
        self.counter += 1

    def result(self) -> float:
        """ Return metric value"""
        return self.wer / self.counter