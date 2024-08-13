import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):

    @abstractmethod
    def forward(self, x):
        return

    @abstractmethod
    def backward(self, x):
        return

    def parameters(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(Module):

    def __init__(self, *modules) -> None:
        super().__init__()
        self.modules: list[Module] = []
        for module in modules:
            if not isinstance(module, Module):
                raise TypeError(f"Expected instance of Module, got {type(module).__name__}")
            self.modules.append(module)

    def append(self, module: Module):
        if not isinstance(module, Module):
            raise TypeError(f"Expected instance of Module, got {type(module).__name__}")
        self.modules.append(module)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, gradient):
        for module in reversed(self.modules):
            gradient = module.backward(gradient)
        return gradient

    def clear_gradients(self):
        for module in self.modules:
            if hasattr(module, "clear_gradients"):
                module.clear_gradients()

    def parameters(self):
        param = {}
        for m in self.modules:
            param.update(m.parameters())
        return param

    def __repr__(self):
        return f"Sequential({'_'.join([module.__repr__() for module in self.modules])})"

    def __str__(self):
        return f"Sequential({'_'.join([module.__str__() for module in self.modules])})"


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weights = np.random.randn(in_dim, out_dim)
        self.bias = np.random.randn(1, out_dim)
        self.weights_gradients = np.zeros((in_dim, out_dim))
        self.bias_gradients = np.zeros((1, out_dim))
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, gradient):
        self.weights_gradients = np.dot(self.input.T, gradient)
        self.bias_gradients = np.sum(gradient, axis=0, keepdims=True)
        grad_input = np.dot(gradient, self.weights.T)
        return grad_input

    def parameters(self):
        return {
            f"linear_weights_{id(self)}": self.weights,
            f"linear_bias_{id(self)}": self.bias
        }

    def clear_gradients(self):
        self.weights_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.bias)

    def __repr__(self):
        return f"Linear_{self.weights.shape[0]}_{self.weights.shape[1]}_{id(self)}"

    def __str__(self):
        return f"Linear({self.weights.shape[0]}, {self.weights.shape[1]})"
