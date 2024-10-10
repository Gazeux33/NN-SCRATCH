import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        return

    @abstractmethod
    def backward(self, y_pred, y_true):
        return

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


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
        y = y.astype(int)
        batch_size = y_pred.shape[0]
        y = y.flatten()  # Aplatir pour l'indexation
        probs = y_pred[np.arange(batch_size), y]
        loss = -np.mean(np.log(probs + 1e-9))
        return loss

    def backward(self,y_pred, y):
        y = y.astype(int)
        batch_size = y_pred.shape[0]
        grad = y_pred.copy()
        grad[np.arange(batch_size), y.flatten()] -= 1
        grad = grad / batch_size
        return grad


class BCE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
