import unittest
import numpy as np
from nn.Loss import MSE, CrossEntropy, BCE


class TestLossFunctions(unittest.TestCase):

    def test_mse_forward(self):
        mse = MSE()
        y_pred = np.array([0.5, 0.6, 0.7])
        y_true = np.array([1.0, 0.0, 0.0])
        expected_loss = np.mean((y_pred - y_true) ** 2)
        self.assertAlmostEqual(mse.forward(y_pred, y_true), expected_loss)

    def test_mse_backward(self):
        mse = MSE()
        y_pred = np.array([0.5, 0.6, 0.7])
        y_true = np.array([1.0, 0.0, 0.0])
        expected_grad = 2 * (y_pred - y_true) / y_true.size
        np.testing.assert_array_almost_equal(mse.backward(y_pred, y_true), expected_grad)

    def test_crossentropy_forward(self):
        cross_entropy = CrossEntropy()
        y_pred = np.array([[0.25, 0.25, 0.5]])
        y_true = np.array([[0, 0, 1]])
        expected_loss = -np.sum(y_true * np.log(y_pred))
        self.assertAlmostEqual(cross_entropy.forward(y_pred, y_true), expected_loss)

    def test_crossentropy_backward(self):
        cross_entropy = CrossEntropy()
        y_pred = np.array([[0.25, 0.25, 0.5]])
        y_true = np.array([[0, 0, 1]])
        expected_grad = -y_true / y_pred
        np.testing.assert_array_almost_equal(cross_entropy.backward(y_pred, y_true), expected_grad)

    def test_bce_forward(self):
        bce = BCE()
        y_pred = np.array([0.8, 0.4, 0.3])
        y_true = np.array([1.0, 0.0, 1.0])
        expected_loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        self.assertAlmostEqual(bce.forward(y_pred, y_true), expected_loss)

    def test_bce_backward(self):
        bce = BCE()
        y_pred = np.array([0.8, 0.4, 0.3])
        y_true = np.array([1.0, 0.0, 1.0])
        expected_grad = -y_true / y_pred + (1 - y_true) / (1 - y_pred)
        np.testing.assert_array_almost_equal(bce.backward(y_pred, y_true), expected_grad)


if __name__ == '__main__':
    unittest.main()
