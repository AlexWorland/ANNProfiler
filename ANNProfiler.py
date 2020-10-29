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
from FinalProject.MNISTDataset import MNIST as MNISTData
from FinalProject.NeuralNetwork import NeuralNetwork as NN

numHiddenLayers = 0
numNeurons = 0
numEpochs = 0
activationFunctionMap = {0: "Rectified Linear (Recommended)",
                         1: "Linear",
                         2: "Exponential Linear Unit",
                         3: "Exponential Activation",
                         4: "Sigmoid",
                         5: "Hard Sigmoid",
                         6: "Scaled Exponential Linear Unit",
                         7: "Softmax",
                         8: "Softplus",
                         9: "Softsign",
                         10: "Swish"}


def main():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    welcomeUser()
    userInfo = getUserInfo(activationFunctionMap)
    mnistData = MNISTData()
    model = tensorFlow.keras.models.Sequential()
    model = intializeModel(userInfo, model)
    model = compileModel(model)
    model = trainModel(model, mnistData)
    foo = "foo"
    print(model)


def welcomeUser():
    print("Hello! Welcome to Alex Worland's Artificial Neural Network Profiler!")
    print("This tool profiles a simple ANN that can predict the value of handwritten digits 0-9")


def getUserInfo(activationFunctionMap):
    numHiddenLayers = int(input("How many hidden layers would you like the network to have? : "))
    numNeurons = int(input("How many neurons would you like those layers to have? : "))
    numEpochs = int(input("How many epochs would you like to train the network? : "))
    print("What activation function would you like to use for those neurons?")
    for i in range(len(activationFunctionMap)):
        print(activationFunctionMap.get(i), ":", i)
    activationFunction = int(input("Please enter the activation function's number: "))
    return [numHiddenLayers, numNeurons, numEpochs, activationFunction]


def intializeModel(userInfo, model):
    numHiddenLayers = userInfo[0]
    numNeurons = userInfo[1]
    numEpochs = userInfo[2]
    # TODO: Might give trouble. May need to reverse map. Will also need a map that maps full activation function
    #   name to shorthand. Could use activations.deserialize instead
    activationFunction = activationFunctionMap.get(userInfo[3])
    model = NN(numHiddenLayers, numNeurons, numEpochs, activationFunction)
    model.createModel()
    return model


def compileModel(model):
    model.compileModel()
    return model


def trainModel(model, mnistData):
    model.trainModel(mnistData)
    # TODO: Training time information will need to be collected here
    return model


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
   # with tensorFlow.device("/gpu:0"):
   #     main()
    with tensorFlow.device("/cpu:0"):
        main()
