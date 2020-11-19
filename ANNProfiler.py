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
from MNISTDataset import MNIST as MNISTData
from NeuralNetwork import NeuralNetwork as NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

activFuncSelectionMap = {1: "Rectified Linear (Recommended For Hidden Layer(s))",
                         2: "Linear",
                         3: "Exponential Linear Unit",
                         4: "Exponential Activation",
                         5: "Sigmoid",
                         6: "Hard Sigmoid",
                         7: "Scaled Exponential Linear Unit",
                         8: "Softmax (Recommended For Output Layer)",
                         9: "Softplus",
                         10: "Softsign",
                         11: "Swish",
                         12: "Hyperbolic Tangent Activation Function"}
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
actionTypes = {1: "Measure Number of Neurons vs Training Time and Accuracy",
               2: "Measure Number of Hidden Layers vs Training Time and Accuracy",
               3: "Measure Number of Epochs vs Training Time and Accuracy",
               # Last entry should always exit program
               4: "Exit Program"
               # TODO: Could add option to plot epoch training time individually
               }
xAxisLabels = {
    1: "Number of Neurons Per Hidden Layer",
    2: "Number of Hidden Layers Per Network",
    3: "Number of Training Epochs Per Network"
    # TODO: Consider adding activation function option
}
patterns = {
    1: "Linear Function",
    2: "Exponential Function (Warning: This can grow very fast. Please choose a small initial number.)",
    3: "Polynomial Function (Warning: This can grow very fast. Please choose a small initial number.)",
    4: "Manual Entry"
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

        # If the action is the last element of actionTypes, exit the program
        if action == len(actionTypes):
            programExit()

        # Get the networks to measure
        unintializedNetworks, xAxisValues = generateUnintializedNetworks(action)

        # Get device to train with from user
        device = deviceSelectionPrompt()

        # TODO: Consider adding multiple execution modes: option to use multiple processing devices

        # Create networks using user specified device
        networks = createNetworks(unintializedNetworks, device)

        # Confirm that the user is ready to commit to training
        confirmation = confirmationPrompt(
            "Training is about to begin. Depending on your configuration this may take a long time.")

        if confirmation:
            # Initialize the training data
            mnist = initializeMNISTData()

            networks, startTimes, endTimes = trainNetworks(networks, mnist, device)

            accuracies, losses = getNetworkMetrics(networks)

            runtimes = calculateRuntimes(startTimes, endTimes)

            plotResults(xAxisValues, xAxisLabels.get(action), runtimes, accuracies)


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


def getActionFromUser():
    """
    A function that gets the measurement type from the user
    :return: An integer representing the measurement type
    """
    actionSelection = multiSelectPrompt("What would you like to do?", "Please enter the action's number: ", actionTypes)
    return actionSelection


def getNumNetworksFromUser():
    """
    A function that gets the number of networks the user would like to train from the user
    :return: An integer representing the number of networks to train
    """
    numNetworks = inputPrompt("How many networks would you like to measure?: ", int)
    return numNetworks


def getNumHiddenLayersFromUser():
    """
    A function that gets the number of hidden layers the user would like each network to have
    :return: An integer representing the number of hidden layers in a network
    """
    numHiddenLayers = inputPrompt("How many hidden layers would you like each network to have?: ", int)
    return numHiddenLayers


def getNumEpochsFromUser():
    """
    A function that gets the number of epochs the user would like to train each network
    :return: An integer representing the number of epochs the user wishes to train with
    """
    epochs = inputPrompt("How many epochs would you like to train each network?: ", int)
    return epochs


def getActivFuncFromUser():
    """
    A function that gets the user's activation function selection
    :return: An integer representing the user's activation function selection
    """
    activationFunctionSelection = multiSelectPrompt(
        "What activation function would you like to use for the hidden layers' neurons?",
        "Please enter the function's number: ",
        activFuncSelectionMap)
    return activationFunctionSelection


def getNumNeuronsFromUser():
    """
    A function that gets the number of neurons the user would like a network to have
    :return: An integer representing the number of neurons the user would like a network to have
    """
    numNeurons = inputPrompt("How many neurons would you like each hidden layer to have?: ", int)
    return numNeurons


def getPatternTypeFromUser(variable):
    """
    A function that gets the pattern that the user would like to use to modify their measurement type
    :param variable: A string holding the word associated with the user's measurement type
    :return: An integer representing the user's pattern selection
    """
    patternType = multiSelectPrompt(
        "How would you like to alter the number of " + variable + " in each network?",
        "Please enter the pattern's number: ",
        patterns
    )
    return patternType


def measureNeurons():
    """
    A function that gets the user's specifications and networks that vary by neuron amount
    :return: A list containing the unintialized neural networks to be measured
    """
    layerSizes = []
    networks = []
    print()
    print("**********************************************************************")

    # Get the number of networks the user would like to train
    numNetworks = getNumNetworksFromUser()

    # Get the number of hidden layers from the user
    numHiddenLayers = getNumHiddenLayersFromUser()

    # Get the pattern type from the user
    patternType = getPatternTypeFromUser("neurons per hidden layer")

    # Get the values that represent the number of neurons in each hidden layer of each network from the user
    layerSizes = getValuesFromPattern(patternType, numNetworks)

    # Get the number of epochs the user would like to train the networks for
    numEpochs = getNumEpochsFromUser()

    # TODO: Refactor this into the getActivFuncFromUser() function
    # Get the hidden layer neurons' activation function from the user
    activationFunctionSelection = getActivFuncFromUser()

    # Get the string representation of the user's selection
    activationFunction = activFuncSelectionMap.get(activationFunctionSelection)

    # Translate the user's activation function selection into a keras compatible string
    activationFunction = activFuncKerasMap.get(activationFunction)

    # Create the networks
    for i in range(numNetworks):
        networks.append(NeuralNetwork(numHiddenLayers, layerSizes[i], numEpochs, activationFunction))

    return networks, layerSizes


def measureHiddenLayers():
    """
    A function that gets the user's specifications and networks that vary by number of hidden layers
    :return: A list containing the unintialized neural networks to be measured
    """
    numHiddenLayers = []
    networks = []
    print()
    print("**********************************************************************")

    # Get the number of networks the user would like to train
    numNetworks = getNumNetworksFromUser()

    # Get the pattern type the user would like to use
    patternType = getPatternTypeFromUser("hidden layers")

    # Get the number of hidden layers for each network based on the user specified pattern
    numHiddenLayers = getValuesFromPattern(patternType, numNetworks)

    # Get the number of neurons the user would like each hidden layer to use
    numNeurons = getNumNeuronsFromUser()

    # Get the number of epochs the user would like each network to train for
    numEpochs = getNumEpochsFromUser()

    # Get the user's activation function selection
    activationFunctionSelection = getActivFuncFromUser()

    # Get the string representation of the user's selection
    activationFunction = activFuncSelectionMap.get(activationFunctionSelection)

    # Translate the user's activation function into a keras compatible version
    activationFunction = activFuncKerasMap.get(activationFunction)

    # Create the uninitialized networks
    for i in range(numNetworks):
        networks.append(NeuralNetwork(numHiddenLayers[i], numNeurons, numEpochs, activationFunction))

    return networks, numHiddenLayers


def measureEpochs():
    """
    A function that gets the user's specifications and networks that vary by number epochs of training
    :return: A list containing the unintialized neural networks to be measured
    """
    numEpochs = []
    networks = []
    print()
    print("**********************************************************************")

    # Get the number of networks the user would like to train
    numNetworks = getNumNetworksFromUser()

    # Get the type of pattern the user would like to use
    patternType = getPatternTypeFromUser("training epochs")

    # Get the training epochs from the pattern the user specified
    numEpochs = getValuesFromPattern(patternType, numNetworks)

    # Get the number of hidden layers the user would like each network to have
    numHiddenLayers = getNumHiddenLayersFromUser()

    # Get the number of neurons the user would like each hidden layer to have
    numNeurons = getNumNeuronsFromUser()

    # Get the user's activation function selection
    activationFunctionSelection = getActivFuncFromUser()

    # Get the string representation of the user's selection
    activationFunction = activFuncSelectionMap.get(activationFunctionSelection)

    # Translate the user's activation function into a keras compatible version
    activationFunction = activFuncKerasMap.get(activationFunction)

    # Create the uninitialized networks
    for i in range(numNetworks):
        networks.append(NeuralNetwork(numHiddenLayers, numNeurons, numEpochs[i], activationFunction))

    return networks, numEpochs


def getValuesFromPattern(patternType, patternSize):
    """
    A function that gets the values from the selected pattern type
    :param patternType: An integer representing the user's pattern type
    :param patternSize: An integer representing how many values the user would like to get
    :return: A list of integers representing the values the user wanted
    """
    # A switch of functions that can be selected based on the pattern type
    switch = {
        1: linearFunction,
        2: exponentialFunction,
        3: polynomialFunction,
        4: manualEntry
    }
    # Get the function that the user specified
    patternFunction = switch.get(patternType)
    # Get the values the user wanted from the function in the switch
    pattern = patternFunction(patternSize)
    return pattern


def linearFunction(patternSize):
    """
    A function that builds a linear function that generates the user specified pattern of values
    :param patternSize: An integer representing the number of values to calculate
    :return: A list of integer values
    """
    values = []

    # Loop to allow the user to confirm their selections
    flag = True
    while flag:
        # Get user specified linear function
        while flag:
            print()
            print("A linear function will be represented as \"y = mx\"")
            coefficient = inputPrompt("What should the value of m be?: ", int)
            variableStart = inputPrompt("What should the starting value of x be? : ", int)
            function = "y = " + str(coefficient) + "x"
            print()
            print("Function will be:", function)

            # Confirm the function with user
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        # Reset the loop flag
        flag = True

        # Calculate the values based on the function the user specified
        for i in range(patternSize):
            value = coefficient * (variableStart + i)
            # Ensure that no value goes below 1 as only positive integers make sense
            if value <= 0:
                valueError(value, variableStart + i, function)
                flag = False
                break
            values.append(value)

        # If the values all check out, display them to the user and confirm
        if flag:
            print()
            print("Values will be:", values)
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

    return values


def polynomialFunction(patternSize):
    """
    A function that builds a polynomial function that generates the user specified pattern of values
    :param patternSize: An integer representing the number of values to calculate
    :return: A list of integer values
    """
    values = []

    # Loop to allow the user to confirm their selections
    flag = True
    while flag:
        # Get user specified polynomial function
        while flag:
            # TODO: Consider adding a 'b' offset to the specification options
            print()
            print("A polynomial function will be represented as \"y = x^n\"")
            power = inputPrompt("What should the value of n be?", int)
            variableStart = inputPrompt("What should the starting value of x be?", int)
            function = "y = x^" + str(power)
            print()
            print(function)

            # Confirm that the function is correct
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        # Reset loop flag
        flag = True

        # Calculate the values of the function
        for i in range(patternSize):
            value = (variableStart + i) ** power
            # Ensure that no value equals zero, if it does, break and start the loop over
            if value <= 0:
                valueError(value, variableStart + i, function)
                flag = False
                break
            values.append(value)

        # If the values all check out, display them and confirm with user
        if flag:
            print()
            print("Values will be:", values)
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

    return values


def exponentialFunction(patternSize):
    """
    A function that builds an exponential function that generates the user specified pattern of values
    :param patternSize: An integer representing the number of values to calculate
    :return: A list of integer values
    """
    values = []

    # Loop to allow the user to confirm their selections
    flag = True
    while flag:
        # Get user specified polynomial function
        while flag:
            print()
            print("An exponential function will be represented as \"y = n^x\"")
            constant = inputPrompt("What should the value of n be? : ", int)
            variableStart = inputPrompt("What should the starting value of x be? : ", int)
            function = "y = " + str(constant) + "^x"
            print()
            print(function)

            # Confirm function configuration with user
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

        # Reset loop flag
        flag = True

        # Calculate the values
        for i in range(patternSize):
            value = constant ** (variableStart + i)

            # Ensure that no value is less than or equal to zero as those values dont make sense
            if value <= 0:
                valueError(value, variableStart + i, function)
                flag = False
                break
            value.append(value)

        # If all values check out, display values then confirm with user
        if flag:
            print()
            print("Values will be:", values)

            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

    return values


def manualEntry(patternSize):
    """
    A function that builds a set of values that are manually entered by the user
    :param patternSize: An integer representing the number of values to calculate
    :return: A list of integer values
    """
    values = []
    # TODO: Add a way to manually change values without having to start over

    # Loop to allow the user to confirm their selections
    flag = True
    while flag:
        for i in range(patternSize):
            value = inputPrompt("Please enter a value for x = " + str(i), int)

            # Check to make sure the user doesnt enter a non positive integer
            if value <= 0:
                valueError(value, i, "Manual Entry")
                flag = False
                break
            values.append(value)
        # If all values check out, display and confirm
        if flag:
            print()
            print("Values will be:", values)
            confirmation = confirmationPrompt()
            if confirmation:
                flag = False

    return values


def inputPrompt(message, typeVar):
    """
    A function that displays a prompt that the user must respond too
    :param message: The prompt the user must answer
    :param typeVar: The type of answer the user must give
    :return: The value that the user responded with
    """

    # Loop to allow checking for the correct type of value
    flag = True
    value = None
    while flag:
        # If the user enters a response not of type "typeVar", we will catch a Value or Type error and display the error
        try:
            print()
            value = input(message)
            value = typeVar(value)
            flag = False

            # Ensure that the value is not less than 1, otherwise the program wont know how to make sense of it
            if typeVar == int:
                if value < 1:
                    inputError(value)
                    flag = True
        except ValueError:
            inputError(value)
        except TypeError:
            inputError(value)

    return value


def multiSelectPrompt(message, query, lst):
    """
    A function that displays a message and a prompt that the user must repsond to. The response must be one of the
    displayed options
    :param message: The message to be displayed
    :param query: A query to be made of the user
    :param lst: A list or dictionary of options the user must select from
    :return: An integer representing the user's selection
    """

    # Loop to allow for error checking
    flag = True
    selection = None
    while flag:
        print()
        print(message)
        print()

        # List can be a list or a dictionary, which both have different access functions
        if type(lst) == dict:
            for i in range(1, len(lst) + 1):
                print(i, ":", lst.get(i))
        # If type(lst) is not a dictionary, it must be a list
        else:
            for i in range(len(lst)):
                print(i+1, ":", lst[i])

        # Get the user's selection
        selection = inputPrompt(query, int)

        # Check if that selection is within range
        if selection >= len(lst) + 1 or selection < 1:
            inputError(selection)
        else:
            flag = False

    return selection


def deviceSelectionPrompt():
    """
    A function that displays the possible training devices tensorflow has access to and asks the user which one to use
    :return: The device object that the user selected
    """
    devices = device_lib.list_local_devices()

    # Loop to allow error checking
    confirmed = False
    while not confirmed:
        # This is -1 because the global behavior was changed so that multiSelect is 1 indexed. This works with
        # everything except this function. TODO: Fix this later
        deviceSelection = multiSelectPrompt(
            "What device would you like to train the networks with?",
            "Please select the device's number: ",
            devices) - 1
        print()
        print("You have selected device:", devices[deviceSelection].name)
        confirmed = confirmationPrompt()

    return devices[deviceSelection]


def confirmationPrompt(*args):
    """
    A function that asks the user if a preceding message should be confirmed
    :param args: Optional arguments representing additional messages to be printed
    :return: A boolean representing whether or not the user confirmed the message
    """
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
    """
    A function that informs the user that a calculated value was invalid
    :param value: The value that is invalid
    :param variable: The variable that caused the error
    :param function: The function that the variable was evaluated with
    :return:
    """
    print()
    print("**********************************************************************")
    print()
    print("Error:", function, "at x =", variable, "equals", value, "which is out of bounds.")
    print()
    print("Please ensure that all values are greater than zero.")
    print()
    print("**********************************************************************")


def generateUnintializedNetworks(measurement):
    """
    A function that generates unintialized networks based on the user's measurement specification
    :param measurement: An integer representing the measurement the user selected
    :return: Returns the networks that the switch generated
    """
    switch = {
        1: measureNeurons,
        2: measureHiddenLayers,
        3: measureEpochs,
    }
    func = switch.get(measurement)
    networks = func()
    return networks


def initializeMNISTData():
    """
    A function that initializes the MNIST dataset to be used
    :return: The MNISTData object that was initialized
    """
    # Initialize the MNIST dataset
    print()
    print("Initializing MNIST dataset...")
    mnistData = MNISTData()
    return mnistData


def createNetworks(networks, device):
    """
    A function that initializes and compiles the networks to be trained
    :param networks: A list of networks to be initialized and compiled
    :param device: The device with which the creationg and compilation should be used with
    :return: The initialized and compiled networks
    """
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
    """
    A function that trains the networks
    :param networks: The networks to be measured and trained
    :param mnistData: The dataset the networks are to be trained on
    :param device: The device to train the networks with
    :return: Returns the networks and their start and end times
    """
    startTimes = []
    endTimes = []
    print()
    print("Training networks...")
    with tensorFlow.device(device.name):
        for i in range(len(networks)):
            print()
            print("Training network", i + 1, "of", len(networks), "...")

            # Make note of the start time for training
            startTimes.append(time.perf_counter())
            # Train the model
            trainModel(networks[i], mnistData)
            # Make note of the end time for training
            endTimes.append(time.perf_counter())

    return networks, startTimes, endTimes


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


def getNetworkMetrics(networks):
    """
    A function that extracts the metric data from the networks
    :param networks: The trained networks to extract the metric data from
    :return: A list of accurcies and a list of losses
    """
    # metrics[0] is loss and metrics[1] is accuracy. Each is found by .result().numpy().item()
    accuracies = []
    losses = []
    for network in networks:
        # Accuracy data is stored in network.model.metrics[1].result().numpy().item()
        accuracies.append(network.model.metrics[1].result().numpy().item() * 100)
        # Loss data is stored in network.model.metrics[1].result().numpy().item()
        losses.append(network.model.metrics[0].result().numpy().item() * 100)

    return accuracies, losses


def calculateRuntimes(startTimes, endTimes):
    """
    A function that calculates the total runtime of a network's training
    :param startTimes: A list of start times
    :param endTimes: A list of end times
    :return:
    """
    runTimes = []
    for i in range(len(startTimes)):
        runTimes.append(endTimes[i] - startTimes[i])

    return runTimes


def plotResults(xAxisValues, xAxisName, trainingTimes, trainingAccuracies):
    """
    A function that plots and displays the results of the training
    :param xAxisValues: Values from 1 - the number of networks trained
    :param xAxisName: The name of the x axis
    :param trainingTimes: A list containing the training duration for each network
    :param trainingAccuracies: A list containing the accuracy percentage for each network
    :return:
    """
    # TODO: Add ability to save plot to disk
    # Loop to allow the user to access the plot more than once
    flag = True
    while flag:
        print()
        print("Plotting results...")

        # Initialize the plots
        fig, ax1 = plt.subplots()

        # Plotting parameters for plotting training duration
        color = 'tab:red'
        ax1.set_xlabel(xAxisName)
        ax1.set_ylabel('training time (seconds)', color=color)
        ax1.set_xticks(xAxisValues)
        # Ensure that the y axis only goes to two decimal points
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # Plot scatter as well as normal plot to get a scatter plot with lines connecting each point
        ax1.scatter(xAxisValues, trainingTimes, color=color)
        ax1.plot(xAxisValues, trainingTimes, color=color)
        # Set the ticks to appear the same color
        ax1.tick_params(axis='y', labelcolor=color)

        # Set up the second plot to share the same x axis as the first plot
        ax2 = ax1.twinx()

        # Plotting parameters for plotting accuracy percentage
        color = 'tab:blue'
        ax2.set_ylabel('training accuracy (% accurate)', color=color)
        # Ensure that the y axis only goes to two decimal points
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # Plot scatter as well as normal plot to get a scatter plot with lines connecting each point
        ax2.scatter(xAxisValues, trainingAccuracies, color=color)
        ax2.plot(xAxisValues, trainingAccuracies, color=color)
        # Set the ticks to appear the same color
        ax2.tick_params(axis='y', labelcolor=color)

        # TODO: investigate what fig.tight_layout() does
        # Show the plot with a grid
        plt.grid()
        plt.show()

        # Main menu confirmation, if user not prepared to go back, plot the data again
        confirmation = confirmationPrompt("Program will now return to main menu.")
        if confirmation:
            flag = False
        else:
            flag = True


def programExit():
    """
    A function that thanks the user and gracefully exits the program.
    :return:
    """
    print()
    print("**********************************************************************")
    print()
    print("Thank you for using Alexandra Worland's Artificial Neural Network Profiler.")
    print()
    print("Goodbye!")
    print()
    print("**********************************************************************")
    exit()


if __name__ == '__main__':
    main()
