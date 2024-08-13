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
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

    def backward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y_true / y_pred


class BCE(Loss):
    def forward(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)



