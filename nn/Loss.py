import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        return

    @abstractmethod
    def backward(self, y_pred, y_true):
        return


class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.y_pred = None
        self.y = None

    def forward(self, y_pred, y):
        num_classes = y_pred.shape[1]
        y_one_hot = np.zeros((y.size, num_classes))
        y_one_hot[np.arange(y.size), y] = 1
        self.y_pred = y_pred
        self.y = y_one_hot
        loss = -np.sum(y_one_hot * np.log(y_pred + 1e-9)) / y.shape[0]
        return loss

    @staticmethod
    def backward(y_pred, y):
        num_classes = y_pred.shape[1]
        y_one_hot = np.zeros((y.size, num_classes))
        y_one_hot[np.arange(y.size), y] = 1
        return (y_pred - y_one_hot) / y.shape[0]


class BCE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
