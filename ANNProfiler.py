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
import tensorflow as tensorFlow
from tensorflow.python.client import device_lib
import time
from FinalProject.MNISTDataset import MNIST as MNISTData
from FinalProject.NeuralNetwork import NeuralNetwork as NeuralNetwork

numHiddenLayers = 0
numNeurons = 0
numEpochs = 0
activFuncSelectionMap = {0: "Rectified Linear (Recommended For Hidden Layer)",
                         1: "Linear",
                         2: "Exponential Linear Unit",
                         3: "Exponential Activation",
                         4: "Sigmoid",
                         5: "Hard Sigmoid",
                         6: "Scaled Exponential Linear Unit",
                         7: "Softmax (Recommended For Output Layer",
                         8: "Softplus",
                         9: "Softsign",
                         10: "Swish",
                         11: "Hyperbolic Tangent Activation Function"}
activFuncKerasMap = {"Rectified Linear (Recommended For Hidden Layer)": 'relu',
                     "Linear": 'linear',
                     "Exponential Linear Unit": 'elu',
                     "Exponential Activation": 'exponential',
                     "Sigmoid": 'sigmoid',
                     "Hard Sigmoid": 'hard_sigmoid',
                     "Scaled Exponential Linear Unit": 'selu',
                     "Softmax (Recommended For Output Layer)": 'softmax',
                     "Softplus": 'softplus',
                     "Softsign": 'softsign',
                     "Swish": 'swish',
                     "Hyperbolic Tangent Activation Function": 'tanh'}


def main():
    # Display the welcome prompt
    welcomeUser()
    # Get the number of hidden layers, neurons, and epochs from user
    userInfo = getUserInfo(activFuncSelectionMap)

    with tensorFlow.device(userInfo[4]):
        print("foo")
        # Initialize the MNIST dataset
        mnistData = MNISTData()
        # Create a new keras Sequential model
        print("Creating Model...")
        model = tensorFlow.keras.models.Sequential()
        # Initialize the model with the user data
        print("Initializing Model...")
        model = intializeModel(userInfo, model)
        # Compile the model
        print("Compiling Model...")
        model = compileModel(model)
        # Train the model with the MNIST dataset
        print("Training Model...")
        print("Beginning Training with", userInfo[2], "epochs:")
        startTime = time.perf_counter()
        model = trainModel(model, mnistData)
        endTime = time.perf_counter()
        totalTrainingTime = endTime - startTime
        print(totalTrainingTime, "seconds.")
        # TODO: Display training metrics


def welcomeUser():
    """
    A function that displays a welcome message
    :return:
    """
    print("Hello! Welcome to Alex Worland's Artificial Neural Network Profiler!")
    print("This tool profiles a simple ANN that can predict the value of handwritten digits 0-9")


def getUserInfo(activationFunctionMap):
    """
    A function that prompts and retrieves user input
    :param activationFunctionMap:
    :return:
    """
    # TODO: Try/Catch for input mismatch
    flag = True
    while flag:
        try:
            numHiddenLayers = int(input("How many hidden layers would you like the network to have? : "))
            print()
            numNeurons = int(input("How many neurons would you like those layers to have? : "))
            print()
            numEpochs = int(input("How many epochs would you like to train the network? : "))
            print()
            print("What activation function would you like to use for those neurons?")
            print()
            for i in range(len(activationFunctionMap)):
                print(i, " : ", activationFunctionMap.get(i))
            print()
            activationFunction = int(input("Please enter the activation function's number: "))
            print()
            print("What device would you like to train this network on?")
            devices = device_lib.list_local_devices()
            for i in range(len(devices)):
                print(i, " : ", devices[i])
            print()
            device = int(input("Please enter the device you would like to train the network on: "))
            device = devices[device].name
            flag = False
        except ValueError:
            print("Please enter a valid number.")

    return [numHiddenLayers, numNeurons, numEpochs, activationFunction, device]


def intializeModel(userInfo, model):
    """
    A function that initializes the model based on the user input
    :param userInfo:
    :param model:
    :return:
    """
    numHiddenLayers = userInfo[0]
    numNeurons = userInfo[1]
    numEpochs = userInfo[2]

    # TODO: Might give trouble. May need to reverse map. Will also need a map that maps full activation function
    #   name to shorthand. Could use activations.deserialize instead

    activationFunction = activFuncSelectionMap.get(userInfo[3])
    # Create new model based on user selections
    model = NeuralNetwork(numHiddenLayers, numNeurons, numEpochs, activationFunction)
    model.createModel()
    return model


def compileModel(model):
    """
    A function that complies the model so that it is read for training
    :param model: the model to be compiled
    :return: the compiled model
    """
    model.compileModel()
    return model


def trainModel(model, mnistData):
    """
    A function that trains the model on the mnist dataset
    :param model: the network model to be trained
    :param mnistData: the data to train the model with
    :return: returns a reference to the model
    """
    model.trainModel(mnistData)
    # TODO: Training time information will need to be collected here
    return model


if __name__ == '__main__':
    # Required environment flags for macos
    # TODO: Look into reason why
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # choice to use gpu or cpu for training
    # with tensorFlow.device("/gpu:0"):
    #    main()
    # with tensorFlow.device("/cpu:0"):
    #     main()
    # TODO: Add capability to choose what device to train with
    main()
