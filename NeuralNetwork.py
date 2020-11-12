"""
Author: Alex Worland
Date: 10/29/2020
File: NeuralNetwork.py
Description:
    This file represents the neural network and all methods associated with retrieving information from it
"""

import tensorflow as tensorFlow
from FinalProject.MNISTDataset import MNIST


class NeuralNetwork:
    """
    A class that represents a neural network model
    """

    # Default Values
    numHiddenLayers = 1
    numNeuronsPerHLayer = 128
    numEpochs = 10
    model = tensorFlow.keras.models.Sequential()
    activationFunctionHidden = 'relu'
    activationFunctionOutput = 'softmax'

    def __init__(self, numHiddenLayers, numNeuronsPerHLayer, trainingEpochs, activationFunction):
        self.numHiddenLayers = numHiddenLayers
        self.numNeuronsPerHLayer = numNeuronsPerHLayer
        self.numEpochs = trainingEpochs
        # TODO: This line requires either a new map of full function names to shorthand or the
        #  activations.deserielize function. For now it will be commented out
        self.createModel()

    def initFromNumHiddenLayers(self, numHiddenLayers):
        self.numHiddenLayers = numHiddenLayers

    def initFromNumNeuronsPerHLayer(self, numNeuronsPerHLayer):
        self.numNeuronsPerHLayer = numNeuronsPerHLayer

    def initFromTrainingEpochs(self, trainingEpochs):
        self.numEpochs = trainingEpochs

    def createModel(self):
        """
        A function that creates the network from the user's parameters
        :return:
        """
        # Input layer. Always the same: 28x28
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
            # TODO: Could be interesting...
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
