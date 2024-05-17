import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        # Initialize weights (including biases) uniformly in the range [0, 1)
        # Was considering separately --> simpler implementation
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        self.input_tensor = None
        self.grad_weights = None
        self._optimizer = None

    def forward(self, input_tensor):
        # Store the input tensor for use in the backward pass
        self.input_tensor = input_tensor

        # Add a bias column of ones to the input tensor
        input_tensor_with_bias = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))

        # Compute the output tensor using the dot product of the input tensor with the combined weights
        output_tensor = np.dot(input_tensor_with_bias, self.weights)

        return output_tensor

    def backward(self, error_tensor):
        # Add a bias column of ones to the input tensor
        input_tensor_with_bias = np.hstack((self.input_tensor, np.ones((self.input_tensor.shape[0], 1))))

        # Calculate gradients for the combined weights
        self.grad_weights = np.dot(input_tensor_with_bias.T, error_tensor)
        print(f"grad weight :   {self.grad_weights}")

        # Calculate error tensor for the previous layer
        error_tensor_previous = np.dot(error_tensor, self.weights[:-1].T)
        print(f"pre_error :  {error_tensor_previous}")

        # If an optimizer is set, update the combined weights using the optimizer's calculate_update method
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.grad_weights)

        return error_tensor_previous

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @property
    def gradient_weights(self):
        # Return the gradient for weights from the combined gradients
        return self.grad_weights

    # @property
    # def gradient_biases(self):
    #     # Return the gradient for biases from the combined gradients
    #     return self.grad_weights[-1:]
    #
    # @property
    # def weights_matrix(self):
    #     # Return the weights part of the combined weights
    #     return self.weights[:-1]
    #
    # @property
    # def biases(self):
    #     # Return the biases part of the combined weights
    #     return self.weights[-1].reshape(1, -1)
