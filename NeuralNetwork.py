"""
Author: Alex Worland
Date: 10/29/2020
File: NeuralNetwork.py
Description:
    This file represents the neural network and all methods associated with retrieving information from it
"""

import tensorflow as tensorFlow

class NeuralNetwork:
    """
    A class that represents a neural network model
    """

    # Default Values
    # TODO: Consider making changeable
    activationFunctionOutput = 'softmax'

    def __init__(self, numHiddenLayers, numNeuronsPerHLayer, trainingEpochs, activationFunction):
        self.model = tensorFlow.keras.models.Sequential()
        self.numHiddenLayers = numHiddenLayers
        self.numNeuronsPerHLayer = numNeuronsPerHLayer
        self.numEpochs = trainingEpochs
        self.activationFunctionHidden = activationFunction

    def createModel(self):
        """
        A function that creates the network from the user's parameters
        :return:
        """
        # Input layer. Always the same size and shape: 28x28
        self.model.add(tensorFlow.keras.layers.Flatten(input_shape=(28, 28)))
        # Add the desired amount of hidden layers, each with the given amount of neurons, each using the given
        # activation function
        for n in range(self.numHiddenLayers):
            self.model.add(
                tensorFlow.keras.layers.Dense(
                    self.numNeuronsPerHLayer,
                    activation=self.activationFunctionHidden
                )
            )
        # Dropout layer helps prevent overfitting
        self.model.add(tensorFlow.keras.layers.Dropout(0.2))
        # Add the output layer. Size is always the same: 10 neurons each representing a base 10 digit
        # User can pick the activation function, though TensorFlow recommends softmax
        self.model.add(tensorFlow.keras.layers.Dense(10))

    def compileModel(self):
        """
        A function that compiles the model
        :return:
        """
        self.model.compile(
            loss=tensorFlow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def trainModel(self, mnistData):
        """
        A function that fits the current model to the mnistData
        :param mnistData: the MNIST dataset
        :return:
        """
        self.model.fit(
            mnistData.trainingData[0],
            mnistData.trainingData[1],
            epochs=self.numEpochs,
        )
