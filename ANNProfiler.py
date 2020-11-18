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

print("\nInitializing...")

# Disable tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import tensorflow as tensorFlow
from tensorflow.python.client import device_lib
import time
from FinalProject.MNISTDataset import MNIST as MNISTData
from FinalProject.NeuralNetwork import NeuralNetwork as NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

activFuncSelectionMap = {0: "Rectified Linear (Recommended For Hidden Layer(s))",
                         1: "Linear",
                         2: "Exponential Linear Unit",
                         3: "Exponential Activation",
                         4: "Sigmoid",
                         5: "Hard Sigmoid",
                         6: "Scaled Exponential Linear Unit",
                         7: "Softmax (Recommended For Output Layer)",
                         8: "Softplus",
                         9: "Softsign",
                         10: "Swish",
                         11: "Hyperbolic Tangent Activation Function"}
activFuncKerasMap = {"Rectified Linear (Recommended For Hidden Layer(s))": 'relu',
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
actionTypes = {0: "Measure Number of Neurons vs Training Time and Accuracy",
               1: "Measure Number of Hidden Layers vs Training Time and Accuracy",
               2: "Measure Number of Epochs vs Training Time and Accuracy",
               # Last entry is always exit program
               3: "Exit Program"
               # TODO: Could add option to plot epoch training time individually
               }
xAxisLabels = {
    0: "Number of Neurons Per Hidden Layer",
    1: "Number of Hidden Layers Per Network",
    2: "Number of Training Epochs Per Network"
}
patterns = {
    0: "Linear Function",
    1: "Exponential Function (Warning: This can grow very fast. Please choose a small initial number.)",
    2: "Polynomial Function (Warning: This can grow very fast. Please choose a small initial number.)",
    3: "Manual Entry"
    # TODO: Consider also: logarithmic, reciprocal, trigonometric, nth root
}


def main():
    """
    Main program function
    :return:
    """
    # Display the welcome prompt
    welcomeUser()

    # Main program loop
    flag = True
    while flag:
        # Get measurement type from the user
        action = getActionFromUser()

        if action == len(actionTypes)-1:
            programExit()

        # Get the networks to measure
        unintializedNetworks, xAxisValues = generateUnintializedNetworks(action)

        # Get device to train with from user
        device = deviceSelectionPrompt()

        # Create the networks using the user specified device
        # TODO: Consider adding multiple execution modes: option to use multiple processing devices

        # Create networks using user specified device
        networks = createNetworks(unintializedNetworks, device)

        ########
        # Debug for models
        models = []
        for network in networks:
            models.append(network.model)
        ########

        flag = confirmationPrompt("Training is about to begin. Depending on your configuration this may take a long time.")
        if flag:
            # Initialize the training data
            mnist = initializeMNISTData()

            networks, startTimes, endTimes, metrics = trainNetworks(networks, mnist, device)

            accuracies, losses = getNetworkMetrics(networks)

            runtimes = calculateRuntimes(startTimes, endTimes)

            plotResults(xAxisValues, xAxisLabels.get(action), runtimes, accuracies)

        else:
            flag = True
    programExit()

def welcomeUser():
    """
    A function that displays a welcome message
    :return:
    """
    print()
    print("**********************************************************************")
    print()
    print("Hello! Welcome to Alexandra Worland's Artificial Neural Network Profiler!")
    print()
    print(
        "This tool profiles a simple ANN that can predict the value of handwritten digits 0-9 using the MNIST dataset.")
    print()
    print("**********************************************************************")

def plotResults(xAxisValues, xAxisName, trainingTimes, trainingAccuracies):
    flag = True
    while flag:
        print()
        print("Plotting results...")

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel(xAxisName)
        ax1.set_ylabel('training time (seconds)')
        ax1.set_xticks(xAxisValues)
        ax1.set_yticks(trainingTimes)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.scatter(xAxisValues, trainingTimes, color=color)
        ax1.plot(xAxisValues, trainingTimes, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('training accuracy (% accurate)')
        ax2.set_yticks(trainingAccuracies)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.scatter(xAxisValues, trainingAccuracies, color=color)
        ax2.plot(xAxisValues, trainingAccuracies, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

        confirmation = confirmationPrompt("Program will now return to main menu.")
        if confirmation:
            flag = False
        else:
            flag = True

# def getUserInfo(activationFunctionMap):
#     """
#     A function that prompts and retrieves user input
#     :param activationFunctionMap:
#     :return:
#     """
#     flag = True
#     while flag:
#         try:
#             numHiddenLayers = inputPrompt("How many hidden layers would you like the network to have?: ", int)
#
#             numNeurons = inputPrompt("How many neurons would you like those layers to have?: ", int)
#
#             numEpochs = inputPrompt("How many epochs would you like to train the network?: ", int)
#
#             print("**********************************************************************")
#
#             print("What activation function would you like to use for those neurons?")
#
#             for i in range(len(activationFunctionMap)):
#                 print(i, ":", activationFunctionMap.get(i))
#
#             activationFunction = int(input("Please enter the activation function's number: "))
#
#             print("**********************************************************************")
#
#             print("What device would you like to train this network on?")
#             devices = device_lib.list_local_devices()
#             for i in range(len(devices)):
#                 print(i, ":", devices[i])
#             device = int(input("Please enter the device you would like to train the network with: "))
#
#             print("**********************************************************************")
#
#             device = devices[device].name
#             flag = False
#         except ValueError:
#             print("Please enter a valid number.")
#
#     return [numHiddenLayers, numNeurons, numEpochs, activationFunction, device]

def getActionFromUser():
    """
    A function that gets the measurement type from the user
    :return: An integer representing the measurement type
    """
    actionSelection = multiSelectPrompt("What would you like to do?", "Please enter the action's number: ", actionTypes)
    return actionSelection

def getNumNetworksFromUser():
    numNetworks = inputPrompt("How many networks would you like to measure?: ", int)
    return numNetworks

def getNumHiddenLayersFromUser():
    numHiddenLayers = inputPrompt("How many hidden layers would you like each network to have?: ", int)
    return numHiddenLayers

def getPatternTypeFromUser(variable):
    patternType = multiSelectPrompt(
        "How would you like to alter the number of " + variable +  " in each network?",
        "Please enter the pattern's number: ",
        patterns
    )
    return patternType

def getNumEpochsFromUser():
    epochs = inputPrompt("How many epochs would you like to train each network?: ", int)
    return epochs

def getActivFuncFromUser():
    activationFunctionSelection = multiSelectPrompt(
        "What activation function would you like to use for the hidden layers' neurons?",
        "Please enter the function's number: ",
        activFuncSelectionMap)
    return activationFunctionSelection

def getNumNeuronsFromUser():
    numNeurons = inputPrompt("How many neurons would you like each hidden layer to have?: ", int)
    return numNeurons

def getValuesFromPattern(patternType, patternSize):
    switch = {
        0: linearFunction,
        1: exponentialFunction,
        2: polynomialFunction,
        3: manualEntry
    }
    patternFunction = switch.get(patternType)
    pattern = patternFunction(patternSize)
    return pattern

def measureNeurons():
    layerSizes = []
    networks = []
    print()
    print("**********************************************************************")

    numNetworks = getNumNetworksFromUser()

    numHiddenLayers = getNumHiddenLayersFromUser()

    patternType = getPatternTypeFromUser("neurons per hidden layer")

    layerSizes = getValuesFromPattern(patternType, numNetworks)

    numEpochs = getNumEpochsFromUser()

    activationFunctionSelection = getActivFuncFromUser()

    activationFunction = activFuncSelectionMap.get(activationFunctionSelection)

    activationFunction = activFuncKerasMap.get(activationFunction)

    for i in range(numNetworks):
        networks.append(NeuralNetwork(numHiddenLayers, layerSizes[i], numEpochs, activationFunction))

    return networks, layerSizes


def measureHiddenLayers():
    numHiddenLayers = []
    networks = []
    print()
    print("**********************************************************************")

    numNetworks = getNumNetworksFromUser()

    patternType = getPatternTypeFromUser("hidden layers")

    numHiddenLayers = getValuesFromPattern(patternType, numNetworks)

    numNeurons = getNumNeuronsFromUser()

    numEpochs = getNumEpochsFromUser()

    activationFunctionSelection = getActivFuncFromUser()

    activationFunction = activFuncKerasMap.get(activationFunctionSelection)

    for i in range(numNetworks):
        networks.append(NeuralNetwork(numHiddenLayers[i], numNeurons, numEpochs, activationFunction))

    return networks, numHiddenLayers

def measureEpochs():
    numEpochs = []
    networks = []
    print()
    print("**********************************************************************")

    numNetworks = getNumNetworksFromUser()

    patternType = getPatternTypeFromUser("training epochs")

    numEpochs = getValuesFromPattern(patternType, numNetworks)

    numHiddenLayers = getNumHiddenLayersFromUser()

    numNeurons = getNumNeuronsFromUser()

    activationFunctionSelection = getActivFuncFromUser()

    activationFunction = activFuncKerasMap.get(activationFunctionSelection)

    for i in range(numNetworks):
        networks.append(NeuralNetwork(numHiddenLayers, numNeurons, numEpochs[i], activationFunction))

    return networks, numEpochs

# def neuronsVsAccuracy():
# def hiddenLayersVsAccuracy():
# def numEpochsVsAccuracy():


def linearFunction(patternSize):
    values = []
    flag = True

    while flag:
        while flag:
            print()
            print("A linear function will be represented as \"y = mx + b\"")

            coefficient = inputPrompt("What should the value of m be?: ", int)

            constant = inputPrompt("What should the value of b be? : ", int)

            variableStart = inputPrompt("What should the starting value of x be? : ", int)


            function = "y = " + str(coefficient) + "x + " + str(constant)
            print()
            print("Function will be:", function)
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False
        flag = True
        for i in range(patternSize):
            value = coefficient * (variableStart + i) + constant
            if value <= 0:
                valueError(value, variableStart + i, function)
                flag = False
                break
            values.append(value)
        if flag:
            print()
            print("Values will be:", values)
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

    return values


def polynomialFunction(patternSize):
    values = []
    flag = True
    while flag:
        while flag:
            print()
            print("A polynomial function will be represented as \"y = x^n\"")


            power = inputPrompt("What should the value of n be?", int)

            variableStart = inputPrompt("What should the starting value of x be?", int)


            print()
            print("Function will be: y = x^" + str(power))


            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        flag = True

        for i in range(patternSize):
            values.append((variableStart + i) ** power)
        print()
        print("Values will be:", values)

        confirmation = confirmationPrompt()
        if confirmation:
            flag = False

    return values


def exponentialFunction(patternSize):
    values = []
    flag = True
    while flag:
        while flag:
            print()
            print("An exponential function will be represented as \"y = n^x\"")


            constant = inputPrompt("What should the value of n be? : ", int)

            variableStart = inputPrompt("What should the starting value of x be? : ", int)

            print()
            print("Function will be: y=", str(constant) + "^x")

            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        flag = True

        for i in range(patternSize):
            values.append(constant ** (variableStart + i))
        print()
        print("Values will be:", values)

        confirmation = confirmationPrompt()
        if confirmation:
            flag = False

    return values


def manualEntry(patternSize):
    values = []
    # TODO: Add a way to manually change values without having to start over
    flag = True
    while flag:
        for i in range(patternSize):
            value = inputPrompt("Please enter a value for x = " + str(i), int)
            values.append(value)
        print()
        print("Values will be:", values)
        confirmation = confirmationPrompt()
        if confirmation:
            flag = False
    return values


def inputPrompt(message, typeVar):
    flag = True
    value = None
    while flag:
        try:
            print()
            value = input(message)
            value = typeVar(value)
            flag = False
        except ValueError:
            inputError(value)
        except TypeError:
            inputError(value)
    return value


def multiSelectPrompt(message, query, lst):
    flag = True
    selection = None
    while flag:
        print()
        print(message)
        print()

        if type(lst) == dict:
            for i in range(len(lst)):
                print(i, ":", lst.get(i))
        else:
            for i in range(len(lst)):
                print(i, ":", lst[i])

        selection = inputPrompt(query, int)
        if selection >= len(lst) or selection < 0:
            inputError(selection)
        else:
            flag = False
    return selection


def deviceSelectionPrompt():
    devices = device_lib.list_local_devices()
    confirmed = False
    while not confirmed:
        deviceSelection = multiSelectPrompt(
            "What device would you like to train the networks with?",
            "Please select the device's number: ",
            devices)
        print()
        print("You have selected device:", devices[deviceSelection].name)
        confirmed = confirmationPrompt()
    return devices[deviceSelection]


def confirmationPrompt(*args):
    flag = True
    confirmationMsg = "Is that ok? (y/n) : "
    for msg in args:
        print()
        print(msg)
    while flag:
        choice = inputPrompt(confirmationMsg, str).lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            inputError(choice)

def inputError(thrownValue):
    """
    A function that informs the user that their input was invalid
    :param thrownValue: The invalid input
    :return:
    """
    print()
    print("**********************************************************************")
    print()
    print("Option: \"", thrownValue, "\" is invalid. Please enter a valid option.")
    print()
    print("**********************************************************************")

def valueError(value, variable, function):
    print()
    print("**********************************************************************")
    print()
    print("Error:", function, "at x =", variable, "equals", value, "which is out of bounds.")
    print()
    print("Please ensure that all values are greater than zero.")
    print()
    print("**********************************************************************")

def initializeMNISTData():
    # Initialize the MNIST dataset
    print()
    print("Initializing MNIST dataset...")
    mnistData = MNISTData()
    return mnistData

def generateUnintializedNetworks(measurement):
    switch = {
        0: measureNeurons,
        1: measureHiddenLayers,
        2: measureEpochs,
    }
    func = switch.get(measurement)
    values = func()
    return values

def createNetworks(networks, device):
    with tensorFlow.device(device.name):
        print()
        print("Creating Networks...")
        for network in networks:
            network.createModel()
        print()
        print("Compiling Networks...")
        for network in networks:
            network.compileModel()
        return networks

def trainNetworks(networks, mnistData, device):
    startTimes = []
    endTimes = []
    metrics = []
    print()
    print("Training networks...")
    with tensorFlow.device(device.name):
        for i in range(len(networks)):
            print()
            print("Training network", i+1, "of", len(networks), "...")
            startTimes.append(time.perf_counter())
            trainModel(networks[i], mnistData)
            endTimes.append(time.perf_counter())
            metrics.append(networks[i].model.metrics)
    return networks, startTimes, endTimes, metrics

def getNetworkMetrics(networks):

    # metrics[0] is loss and metrics[1] is accuracy. Each is found by .result().numpy().item()
    accuracies = []
    losses = []
    for network in networks:
        accuracies.append(network.model.metrics[1].result().numpy().item() * 100)
        losses.append(network.model.metrics[0].result().numpy().item() * 100)
    return accuracies, losses

def initializeModel(userInfo):
    """
    A function that initializes the model based on the user input
    :param userInfo: an array containing relevant information from the user
    :param model: a neural network model created to user specifications
    :return:
    """
    # Get user information from userInfo array
    numHiddenLayers = userInfo[0]
    numNeurons = userInfo[1]
    numEpochs = userInfo[2]
    activationFunctionLonghand = activFuncSelectionMap.get(userInfo[3])
    activationFunctionKeras = activFuncKerasMap.get(activationFunctionLonghand)

    # Create new model based on user selections
    model = NeuralNetwork(numHiddenLayers, numNeurons, numEpochs, activationFunctionKeras)
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


def calculateRuntimes(startTimes, endTimes):
    runTimes = []
    for i in range(len(startTimes)):
        runTimes.append(endTimes[i] - startTimes[i])
    return runTimes

def programExit():
    print()
    print("Thank you for using Alexandra Worland's Artificial Neural Network Profiler.")
    print()
    print("Goodbye!")
    exit()


if __name__ == '__main__':
    main()
