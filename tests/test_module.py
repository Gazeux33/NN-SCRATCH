import unittest
import numpy as np
from nn.Module import Linear, Sequential


class TestLinear(unittest.TestCase):

    def setUp(self):
        self.linear = Linear(10, 20)

    def test_linear_weights_shape(self):
        self.assertEqual(self.linear.weights.shape, (10, 20))

    def test_linear_bias_shape(self):
        self.assertEqual(self.linear.bias.shape, (1, 20))

    def test_linear_wrong_input_shape(self):
        wrong_input = np.random.randn(100, 20)
        with self.assertRaises(ValueError):
            self.linear(wrong_input)

    def test_linear_correct_input_shape(self):
        correct_input = np.random.randn(100, 10)
        output = self.linear(correct_input)
        self.assertEqual(output.shape, (100, 20))


    def test_linear_backward(self):
        input_data = np.random.randn(5, 10)
        output = self.linear(input_data)

        grad_output = np.random.randn(5, 20)

        grad_input = self.linear.backward(grad_output)

        self.assertEqual(self.linear.weights_gradients.shape, (10, 20))
        self.assertEqual(self.linear.bias_gradients.shape, (1, 20))
        self.assertEqual(grad_input.shape, (5, 10))

    def test_clear_gradient(self):
        input_data = np.random.randn(5, 10)
        self.linear(input_data)

        grad_output = np.random.randn(5, 20)
        self.linear.backward(grad_output)

        self.linear.clear_gradients()

        np.testing.assert_array_equal(self.linear.weights_gradients,np.zeros_like(self.linear.weights_gradients))
        np.testing.assert_array_equal(self.linear.bias_gradients, np.zeros_like(self.linear.bias_gradients))


class TestSequential(unittest.TestCase):

    def setUp(self):
        self.seq = Sequential(Linear(10, 20), Linear(20, 100))

    def test_sequential_output_shape(self):
        input_data = np.random.randn(2, 10)
        output = self.seq(input_data)
        self.assertEqual(output.shape, (2, 100))

    def test_sequential_append_layers(self):
        seq = Sequential()
        seq.append(Linear(10, 20))
        seq.append(Linear(20, 100))
        input_data = np.random.randn(2, 10)
        output = seq(input_data)
        self.assertEqual(output.shape, (2, 100))

    def test_sequential_parameters(self):
        params = self.seq.parameters()
        self.assertEqual(type(params), dict)
        self.assertEqual(len(params), 4)

    def test_sequential_backward(self):
        input_data = np.random.randn(5, 10)
        output = self.seq(input_data)

        grad_output = np.random.randn(5, 100)
        grad_input = self.seq.backward(grad_output)

        self.assertEqual(grad_input.shape, (5, 10))

        self.assertEqual(self.seq.modules[0].weights_gradients.shape, (10, 20))
        self.assertEqual(self.seq.modules[0].bias_gradients.shape, (1, 20))
        self.assertEqual(self.seq.modules[1].weights_gradients.shape, (20, 100))
        self.assertEqual(self.seq.modules[1].bias_gradients.shape, (1, 100))

    def test_clear_gradient(self):
        pass


if __name__ == '__main__':
    unittest.main()
