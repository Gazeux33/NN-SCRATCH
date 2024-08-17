from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, model):
        return


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def step(self, model):
        for layer in model.modules:
            if hasattr(layer, "weights") and hasattr(layer, "weights_gradients"):
                layer.weights -= self.lr * layer.weights_gradients
            if hasattr(layer, "bias") and hasattr(layer, "bias_gradients"):
                layer.bias -= self.lr * layer.bias_gradients


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, model):
        self.t += 1
        for i, layer in enumerate(model.modules):
            if hasattr(layer, "weights") and hasattr(layer, "weights_gradients"):
                if i not in self.m:
                    self.m[i] = {"weights": np.zeros_like(layer.weights), "bias": np.zeros_like(layer.bias)}
                    self.v[i] = {"weights": np.zeros_like(layer.weights), "bias": np.zeros_like(layer.bias)}

                self.m[i]["weights"] = self.beta1 * self.m[i]["weights"] + (
                        1 - self.beta1) * layer.weights_gradients
                self.v[i]["weights"] = self.beta2 * self.v[i]["weights"] + (1 - self.beta2) * (
                        layer.weights_gradients ** 2)

                self.m[i]["bias"] = self.beta1 * self.m[i]["bias"] + (1 - self.beta1) * layer.bias_gradients
                self.v[i]["bias"] = self.beta2 * self.v[i]["bias"] + (1 - self.beta2) * (layer.bias_gradients ** 2)

                m_hat_weights = self.m[i]["weights"] / (1 - self.beta1 ** self.t)
                v_hat_weights = self.v[i]["weights"] / (1 - self.beta2 ** self.t)

                m_hat_bias = self.m[i]["bias"] / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v[i]["bias"] / (1 - self.beta2 ** self.t)

                layer.weights -= self.lr * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
                layer.bias -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
