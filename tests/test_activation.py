import unittest
import numpy as np
from PyNetworks.Activation import Relu, Sigmoid, Tanh, Softmax


class TestActivations(unittest.TestCase):

    def test_relu_forward(self):
        relu = Relu()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(relu.forward(x), expected_output)

    def test_relu_backward(self):
        relu = Relu()
        x = np.array([-1.0, 0.0, 1.0])
        relu.forward(x)  # To store the input
        gradient = np.array([1.0, 1.0, 1.0])
        expected_gradient = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(relu.backward(gradient), expected_gradient)

    def test_sigmoid_forward(self):
        sigmoid = Sigmoid()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid.forward(x), expected_output)

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()
        x = np.array([-1.0, 0.0, 1.0])
        output = sigmoid.forward(x)
        gradient = np.array([1.0, 1.0, 1.0])
        expected_gradient = gradient * output * (1 - output)
        np.testing.assert_array_almost_equal(sigmoid.backward(gradient), expected_gradient)

    def test_tanh_forward(self):
        tanh = Tanh()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = np.tanh(x)
        np.testing.assert_array_almost_equal(tanh.forward(x), expected_output)

    def test_tanh_backward(self):
        tanh = Tanh()
        x = np.array([-1.0, 0.0, 1.0])
        output = tanh.forward(x)  # To store the output
        gradient = np.array([1.0, 1.0, 1.0])
        expected_gradient = gradient * (1 - output ** 2)
        np.testing.assert_array_almost_equal(tanh.backward(gradient), expected_gradient)

    def test_softmax_forward(self):
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        expected_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(softmax.forward(x), expected_output)

    def test_softmax_backward(self):
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        softmax.forward(x)
        gradient = np.array([[0.1, 0.2, 0.7]])
        expected_gradient = gradient
        np.testing.assert_array_almost_equal(softmax.backward(gradient), expected_gradient)


if __name__ == '__main__':
    unittest.main()
