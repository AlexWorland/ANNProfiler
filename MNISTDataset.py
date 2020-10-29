"""
Author: Alex Worland
Date: 10/29/2020
File: MNISTDataset.py
Description:
    This file retrieves the MNIST dataset that the NeuralNetwork.py file will use
"""

import tensorflow_datasets as dataSets
import tensorflow as tensorFlow


class MNIST:
    trainingData = ""
    testingData = ""

    def __init__(self):
        """
        Initializes the training and testing data as well as the infoData
        """
        (trainingData, testingData), infoData = dataSets.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True)
        self.trainingData = self.__initTrainingData(trainingData, infoData)
        self.testingData = self.__initTestingData(testingData)

    @staticmethod
    def normalizeImage(image, label):
        """
        A function that normalizes the incoming image data from uint8 to float32
        """
        return tensorFlow.cast(image, tensorFlow.float32) / 255.0, label

    def __initTrainingData(self, trainingData, infoData):
        """
        Builds the training pipeline
        :param trainingData: The training data that has come in from MNIST
        :param infoData: Information about the
        :return:
        """
        # Normalize the training images from uint8 to float32
        trainingData = trainingData.map(
            self.normalizeImage, num_parallel_calls=tensorFlow.data.experimental.AUTOTUNE
        )
        # Cache the training dataset to memory for better shuffling performance
        trainingData = trainingData.cache()
        # Shuffle the data
        trainingData = trainingData.shuffle(infoData.splits['train'].num_examples)
        # After shuffling, batch the data
        trainingData = trainingData.batch(128)
        # Prefetch to improve performance
        trainingData = trainingData.prefetch(tensorFlow.data.experimental.AUTOTUNE)
        return trainingData

    def __initTestingData(self, testingData):
        """
        Builds the testing pipeline
        :param testingData: The testing data that has come in from MNIST
        :return:
        """
        # Normalize testing images from uint8 to float32
        testingData = testingData.map(self.normalizeImage, num_parallel_calls=tensorFlow.data.experimental.AUTOTUNE)
        # Batch the data
        testingData.batch(128)
        # Cache the data to memory
        testingData = testingData.cache()
        # Prefetch to improve performance
        testingData = testingData.prefetch(tensorFlow.data.experimental.AUTOTUNE)
        return testingData