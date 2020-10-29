"""
Author: Alex Worland
Date: 10/29/2020
File: NeuralNetwork.py
Description:
    This file represents the neural network and all methods associated with retrieving information from it
"""


class NeuralNetwork:
    # Default Values
    numHiddenLayers = 1
    numNeuronsPerHLayer = 8
    trainingEpochs = 10

    def __init__(self, numHiddenLayers, numNeuronsPerHLayer, trainingEpochs):
        self.numHiddenLayers = numHiddenLayers
        self.numNeuronsPerHLayer = numNeuronsPerHLayer
        self.trainingEpochs = trainingEpochs

    def initFromNumHiddenLayers(self, numHiddenLayers):
        self.numHiddenLayers = numHiddenLayers

    def initFromNumNeuronsPerHLayer(self, numNeuronsPerHLayer):
        self.numNeuronsPerHLayer = numNeuronsPerHLayer

    def initFromTrainingEpochs(self, trainingEpochs):
        self.trainingEpochs = trainingEpochs

