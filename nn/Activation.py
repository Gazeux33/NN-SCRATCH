from nn.Module import Module
import numpy as np


class Relu(Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient * (self.input > 0)


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient * self.output * (1 - self.output)


class Tanh:
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return np.tanh(x)

    def backward(self, gradient):
        return gradient * (1 - np.tanh(self.input) ** 2)


class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    @staticmethod
    def backward(gradient):
        return gradient



