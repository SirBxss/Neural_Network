# NeuralNetwork.py
import numpy as np
from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer):
        """
        Constructor for NeuralNetwork.

        Parameters:
            optimizer (object): An optimizer object to use for the neural network's layers.
        """
        # Initialize member variables
        self.optimizer = optimizer
        self.loss = []  # List to store loss value for each iteration
        self.layers = []  # List to hold the architecture
        self.data_layer = None  # Placeholder for data layer (input data and labels)
        self.loss_layer = None  # Placeholder for loss layer (special layer providing loss and prediction)

    def forward(self):
        """
        Forward pass using input from the data layer and passing it through all layers of the network.

        Returns:
            output: The output of the last layer (loss layer) of the network.
        """
        # Get input and labels from data layer
        input_tensor, label_tensor = self.data_layer.next()

        # Forward pass through all layers
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # Store the label tensor for backward pass
        self.label_tensor = label_tensor

        output = self.loss_layer.forward(input_tensor, label_tensor)

        # Return the output from the loss layer
        return output

    def backward(self):
        """
        Backward pass starting from the loss layer, passing it the label tensor for the current input
        and propagating it back through the network.
        """
        # Start the backward pass from the loss layer using the stored label tensor
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate the error tensor through all layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        # If the layer is trainable, set the optimizer
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)

        # Append the layer to the list of layers
        self.layers.append(layer)

    def train(self, iterations):
        """
        Train the neural network for a specified number of iterations.

        Parameters:
            iterations (int): The number of iterations to train the network.
        """
        # Loop through the specified number of iterations
        for _ in range(iterations):
            # Perform forward and backward pass
            self.forward()
            self.backward()
            # Calculate and store the loss for the current iteration
            loss_value = self.loss_layer.loss
            self.loss.append(loss_value)

    def test(self, input_tensor):
        """
        Propagate the input tensor through the network and return the prediction of the last layer.

        Parameters:
            input_tensor (ndarray): The input tensor to propagate through the network.

        Returns:
            output_tensor (ndarray): The output tensor from the last layer of the network.
        """
        # Forward pass through all layers using the input tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # Return the output tensor from the last layer
        return input_tensor
