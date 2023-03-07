import torch

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

    def update(self, output: torch.Tensor, target: torch.Tensor):
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
    def __init__(self, name='accuracy') -> None:
        super(Accuracy, self).__init__(name=name)
        self.correct = 0
        self.total = 0

    def update(self, output: torch.Tensor, target: torch.Tensor):
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