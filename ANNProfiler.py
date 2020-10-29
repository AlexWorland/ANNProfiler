"""
Author: Alex Worland
Date: 10/29/2020
File: ANNProfiler.py
Description:
    This is an artificial neural network profiling tool. Using the MNIST data set, a simple ANN is used to
    predict a handwritten digit represented as a 28x28 pixel grayscale image. This tool will profile the
    time it takes to train this network and assess its accuracy.
        The user can specify multiple variables:
            * The number of epochs used during training
            * The number of neurons per hidden layer
            * The number of hidden layers
    The user can also interact with the network directly through the user input mode. The user can draw their
    own digit with their mouse and see the network work visually and see the prediction and accuracy displayed.
"""