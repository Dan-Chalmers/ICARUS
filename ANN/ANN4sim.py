
''' ANN4sim.py '''

# 5 output nodes -> 0, 1, 2, 3, 4,

import numpy as NP
import pandas as PD



def logisticFunction(x):
    '''
    Perform a sigmoid calculation on the parameter

    parameters
    ----------
    x : float
        The multiplication of two values

    returns
    -------
    a : float
        Result of the sigmoid calculation
    '''

    a = 1 / (1 + NP.exp(-x))
    return a


def readData(dir):
    '''
    Read data from the relevant CSV in paralell using dask and combine into a PD DataFrame and then an NP array

    returns
    -------
    labels : NP.ndarray
        An array of the image labels. Size (numPixels, 1)
    trainingData : NP.ndarray
        An array containing the images' pixel data. Size (numImages, numPixels)
    '''

    trainingData = PD.read_csv(dir) # read in CSV in parallel uising dask dataframe
    trainingData = trainingData.to_numpy()
    return trainingData


def forwardPropagate(data, inputHiddenConnection, hiddenOutputConnection):
    '''
    Feeds-forward through the network to multiply the nodes by the respective weights

    Parameters
    ----------
    data : NP.ndarray
        The data to be used as the input values (testing)
    inputHiddenConnection : NP.ndarray
        The mesh of weights between the input layer and the hidden layer
    hiddenOutputConnection : NP.ndarray
        The mesh of weights between the hidden layer and the output layer

    returns
    -------
    outputVals : NP.ndarray
        The five output values (largest of which is the prediction)
    '''

    inputVals = NP.array(data).transpose()
    hiddenLayerInputs = NP.dot(inputHiddenConnection, inputVals)
    hiddenLayerOutputs = logisticFunction(hiddenLayerInputs)
    outputLayerInputs = NP.dot(hiddenOutputConnection, hiddenLayerOutputs)
    outputVals = logisticFunction(outputLayerInputs)
    return outputVals


def validation(data, inputHiddenConnection, hiddenOutputConnection):
    '''
    Run the validation set and calculate final accuracies for each shape

    parameters
    ----------
    data : NP.ndarray
        The validation data set
    inputHiddenConnection : NP.ndarray
        The mesh of weights between the input layer and the hidden layer
    hiddenOutputConnection : NP.ndarray
        The mesh of weights between the hidden layer and the output layer
    labels : NP.ndarray
        The labels for the validation set to check network predcitions

    returns
    -------
    All accuracies as a percentage
    '''

    predTotalArray = []

    for i in range(0, len(data)):
        predArray = forwardPropagate(data[i], inputHiddenConnection, hiddenOutputConnection)
        prediction = max(predArray)
        predictionIdx = -1
        
        for j in range(0, len(predArray)):
            predictionIdx += 1
            if predArray[j] == prediction:
                break

        predTotalArray.append(predictionIdx)

    return predTotalArray


def codeRunner(dir):
    '''
    Run from main() -> boilerplate code to manage the algorithm and run functions
    '''

    trainingData  = readData(dir)
    scaledTrainingData = (NP.asfarray(trainingData[0:]) / 255.0 * 0.99) + 0.01 # Scale each value to match range of sigmoid function

    #!- VALIDATION -!#
    inputHiddenConnection = (PD.read_csv('E:/inputHiddenWeights1000_100_0.0005.csv')).to_numpy()
    hiddenOutputConnection = (PD.read_csv('E:/hiddenOutputWeights1000_100_0.0005.csv')).to_numpy()
    predictions = validation(scaledTrainingData, inputHiddenConnection, hiddenOutputConnection)
    return predictions


