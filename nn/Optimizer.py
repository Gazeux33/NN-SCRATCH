from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def step(self, model):
        return


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model):
        for layer in model.modules:
            if hasattr(layer, "weights") and hasattr(layer, "weights_gradients"):
                layer.weights -= self.lr * layer.weights_gradients
            if hasattr(layer, "bias") and hasattr(layer, "bias_gradients"):
                layer.bias -= self.lr * layer.bias_gradients

