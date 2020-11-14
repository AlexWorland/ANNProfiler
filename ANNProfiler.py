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
import tensorflow as tensorFlow
from tensorflow.python.client import device_lib
import time
from FinalProject.MNISTDataset import MNIST as MNISTData
from FinalProject.NeuralNetwork import NeuralNetwork as NeuralNetwork

numHiddenLayers = 0
numNeurons = 0
numEpochs = 0
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
measurements = {0: "Number of Neurons vs Training Time",
                1: "Number of Hidden Layers vs Training Time",
                2: "Number of Epochs vs Training Time",
                3: "Number of Neurons vs Training Accuracy",
                4: "Number of Hidden Layers vs Training Accuracy",
                5: "Number of Epochs vs Training Accuracy"}
patterns = {
    0: "Linear",
    1: "Exponential (Warning: This can grow very fast. Please choose a small initial number.)",
    2: "Polynomial (Warning: This can grow very fast. Please choose a small initial number.)",
    3: "Manual"
    # TODO: Consider also: logarithmic, reciprocal, trigonometric, nth root
}


def main():
    # Display the welcome prompt
    welcomeUser()

    measurementType = getMeasurementTypeFromUser()

    values = measurementPrompts(measurementType)

    # # Get the number of hidden layers, neurons, and epochs from user
    # userInfo = getUserInfo(activFuncSelectionMap)

    # with tensorFlow.device(userInfo[4]):
    #     # Initialize the MNIST dataset
    #     mnistData = MNISTData()
    #
    #     # Initialize the model with the user data
    #     print("Initializing Model...")
    #     model = intializeModel(userInfo)
    #
    #     # Compile the model
    #     print("Compiling Model...")
    #     model = compileModel(model)
    #
    #     # Train the model with the MNIST dataset
    #     print("Training Model...")
    #     print("Beginning Training with", userInfo[2], "epochs:")
    #     startTime = time.perf_counter()
    #     model = trainModel(model, mnistData)
    #     endTime = time.perf_counter()
    #     totalTrainingTime = endTime - startTime
    #     print(totalTrainingTime, "seconds.")
    #     # TODO: Display training metrics


def welcomeUser():
    """
    A function that displays a welcome message
    :return:
    """
    print("**********************************************************************")
    print()
    print("Hello! Welcome to Alexandra Worland's Artificial Neural Network Profiler!")
    print()
    print(
        "This tool profiles a simple ANN that can predict the value of handwritten digits 0-9 using the MNIST dataset.")
    print()
    print("**********************************************************************")
    print()


def getMeasurementTypeFromUser():
    """
    A function that gets the measurement type from the user
    :return: An integer representing the measurement type
    """

    # TODO: This type of thing happens a lot, maybe refactor it into something that can handle it?
    # Loop to ensure user input is in the correct format
    flag = True
    while flag:
        print()
        print("**********************************************************************")
        while flag:
            try:
                print()
                print("What would you like to measure?")
                print()

                # List the different measurement types
                for i in range(len(measurements)):
                    print(i, ":", measurements.get(i))

                print()
                measurement = input("Please enter the measurement's number: ")
                # Cast input to int
                measurement = int(measurement)

                # Check if the input is within range
                if measurement >= len(measurements) or measurement < 0:
                    flag = True
                    raise ValueError
                else:
                    flag = False
            # If input is invalid, inform user and restart loop
            except ValueError:
                inputError(measurement)

        # Loop to ensure user input is in the correct format
        flag = True
        while flag:
            try:
                print()
                print("You have selected: \"", measurements.get(measurement), "\"")
                print()
                confirmation = input("Is that correct? (y/n): ")

                # Cast input to string
                confirmation = str(confirmation).lower()

                # Check if input is y or n
                if confirmation == 'y':
                    flag = False
                elif confirmation == 'n':
                    flag = True
                else:
                    # If not y or n, throw a value error
                    raise ValueError
                print()
                print("**********************************************************************")
                print()
            except ValueError:
                inputError(confirmation)
    return measurement


def initializeNetworkList(measurement):
    print()
    print("How many networks would you like to measure?")


def measurementPrompts(measurement):
    switch = {
        0: neuronsVsTime,
        # 1: hiddenLayersVsTime,
        # 2: numEpochsVsTime,
        # 3: neuronsVsAccuracy,
        # 4: hiddenLayersVsAccuracy,
        # 5: numEpochsVsAccuracy
    }
    func = switch.get(measurement)
    values = func()
    return values


def neuronsVsTime():
    layerSizes = []
    print("**********************************************************************")
    print()
    numHiddenLayers = input("How many hidden layers would you like each network to have?")
    numHiddenLayers = int(numHiddenLayers)
    print()
    print("How would you like to alter the number of neurons per hidden layer in each network?")
    print()
    for i in range(len(patterns)):
        print(i, ":", patterns.get(i))
    print()
    patternType = input("Please enter the pattern's number: ")
    patternType = int(patternType)
    # print()
    # firstHLayerSize = input("How many neurons would you like your first layer to have? : ")
    # firstHLayerSize = int(firstHLayerSize)
    # print()
    # if patternType == 2:
    #     layerSizes.append(firstHLayerSize)
    #     for i in range(1, numHiddenLayers):
    #         numNeurons = input("How many neurons should network" + str(i + 1) + "/" + str(numHiddenLayers) + "have? : ")
    #         numNeurons = int(numNeurons)
    #         layerSizes.append(numNeurons)
    # else:
    #     layerSizes = patternFunctions(patternType, numHiddenLayers, firstHLayerSize)

    layerSizes = patternFunctions(patternType, numHiddenLayers)


def patternFunctions(patternType, patternSize):
    switch = {
        0: linearFunction,
        1: exponentialFunction,
        2: polynomialFunction,
        3: manualEntry
    }
    patternFunction = switch.get(patternType)
    pattern = patternFunction(patternSize)
    return pattern


def linearFunction(patternSize):
    values = []
    flag = True

    while flag:
        while flag:
            print()
            print("A linear function will be represented as \"y = mx + b\"")
            print()

            coefficient = inputPrompt("What should the value of m be?: ", int)
            print()
            constant = inputPrompt("What should the value of b be? : ", int)
            print()
            variableStart = inputPrompt("What should the starting value of x be? : ", int)
            print()

            print()
            print("Function will be: y =", str(coefficient) + "x + " + str(constant))
            print()

            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        flag = True

        for i in range(patternSize):
            values.append(coefficient * (variableStart + i) + constant)

        print("Values will be", values)

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
            print()

            power = inputPrompt("What should the value of n be?", int)
            print()
            variableStart = inputPrompt("What should the starting value of x be?")
            print()

            print()
            print("Function will be: y = x^" + str(power))
            print()

            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        flag = True

        for i in range(patternSize):
            values.append((variableStart + i) ** power)

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
            print()

            constant = inputPrompt("What should the value of n be? : ")
            print()
            variableStart = inputPrompt("What should the starting value of x be? : ")
            print()

            print("Function will be: y=", str(constant) + "^x")

            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        flag = True

        for i in range(patternSize):
            values.append(constant ** (variableStart + i))

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
            value = input(message)
            value = typeVar(value)
            flag = False
        except ValueError:
            inputError(value)
        except TypeError:
            inputError(value)
    return value


def confirmationPrompt():
    flag = True
    message = "Is that ok? (y/n) : "
    while flag:
        choice = inputPrompt(message, str).lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            inputError(choice)


# def hiddenLayersVsTime():
# def numEpochsVsTime():
# def neuronsVsAccuracy():
# def hiddenLayersVsAccuracy():
# def numEpochsVsAccuracy():

def inputError(thrownValue):
    """
    A function that informs the user that their input was invalid
    :param thrownValue: The invalid input
    :return:
    """
    print()
    print("**********************************************************************")
    print("Option: \"", thrownValue, "\" is invalid. Please enter a valid option.")
    print("**********************************************************************")
    print()


def getUserInfo(activationFunctionMap):
    """
    A function that prompts and retrieves user input
    :param activationFunctionMap:
    :return:
    """
    flag = True
    while flag:
        try:
            numHiddenLayers = int(input("How many hidden layers would you like the network to have?:"))
            print()
            numNeurons = int(input("How many neurons would you like those layers to have?:"))
            print()
            numEpochs = int(input("How many epochs would you like to train the network?:"))
            print()
            print("**********************************************************************")
            print()
            print("What activation function would you like to use for those neurons?")
            print()
            for i in range(len(activationFunctionMap)):
                print(i, ":", activationFunctionMap.get(i))
            print()
            activationFunction = int(input("Please enter the activation function's number: "))
            print()
            print("**********************************************************************")
            print()
            print("What device would you like to train this network on?")
            devices = device_lib.list_local_devices()
            for i in range(len(devices)):
                print(i, ":", devices[i])
            device = int(input("Please enter the device you would like to train the network with: "))
            print()
            print("**********************************************************************")
            print()
            device = devices[device].name
            flag = False
        except ValueError:
            print("Please enter a valid number.")

    return [numHiddenLayers, numNeurons, numEpochs, activationFunction, device]


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


if __name__ == '__main__':
    main()
