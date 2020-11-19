"""
Author: Alex Worland
Date: 10/29/2020
File: MNISTDataset.py
Description:
    This file retrieves the MNIST dataset that the NeuralNetwork.py file will use
"""

import tensorflow as tensorFlow


class MNIST:
    trainingData = ""
    validationData = ""

    def __init__(self):
        """
        Initializes the training and testing data as well as the infoData
        """
        mnist = tensorFlow.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Convert samples from int to float
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.trainingData = (x_train, y_train)
        self.validationData = (x_test, y_test)
