import torch

class Metric:
    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self):
        # reset metric state to initial values and return metric value
        self.__init__()

    def update(self, output, target):
        # update metric state with new data
        pass

    def result(self):
        # return metric value
        pass

class Accuracy(Metric):
    def __init__(self, name='accuracy') -> None:
        super(Accuracy, self).__init__(name=name)
        self.correct = 0
        self.total = 0

    def update(self, output, target):
        _, predicted = torch.max(output.data, 1)
        self.total += target.size(0)
        self.correct += (predicted == target).sum().item()

    def result(self):
        return self.correct / self.total