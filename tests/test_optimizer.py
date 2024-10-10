import unittest
import numpy as np
from PyNetworks.Optimizer import SGD
from PyNetworks.Module import Sequential, Linear
from PyNetworks.Loss import MSE


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.model = Sequential(
            Linear(2, 10),
            Linear(10, 1)
        )

        self.test_labels = np.random.random((100, 2))
        self.test_targets = np.random.random((100, 1))

        self.loss = MSE()

    def test_SGD(self):
        self.sgd = SGD()

        # forward pass
        output = self.model(self.test_labels)

        loss = self.loss.forward(output, self.test_targets)
        print(loss)

        gradient = self.loss.backward(output, self.test_targets)
        self.model.backward(gradient)


if __name__ == '__main__':
    unittest.main()
