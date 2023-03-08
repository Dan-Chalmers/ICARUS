
''' CNN.py '''

# 5 output nodes -> 0, 1, 2, 3, 4,

import numpy as NP
import pandas as PD
import dask.dataframe as Ddf

from itertools import chain as CHN


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


def readData(sheet):
    '''
    Read data from the relevant CSV in paralell using dask and combine into a PD DataFrame and then an NP array

    returns
    -------
    labels : NP.ndarray
        An array of the image labels. Size (numPixels, 1)
    trainingData : NP.ndarray
        An array containing the images' pixel data. Size (numImages, numPixels)
    '''

    labels = ((PD.read_excel('E://trainingLabels.xlsx', sheet, header=None)).drop(index=0)).to_numpy()
    #labels = labels.to_numpy()
    labelsTest = (PD.read_excel('E://validationLabels.xlsx', sheet, header=None)).drop(index=0)
    labelsTest = list(CHN.from_iterable(labelsTest.values.tolist()))
    labelsTest = [int(x) for x in labelsTest]
    print (labelsTest)
    trainingData = Ddf.read_csv('E://compressed5000PixelData.csv') # read in CSV in parallel uising dask dataframe
    trainingData = trainingData.compute() # Join all partitioned dask dataframes in a pandas df
    trainingData = trainingData.to_numpy()
    testData = PD.read_csv('E://compressed5000ValidationPixelData.csv')
    print (testData.shape[0])
    testData = testData.to_numpy()
    print (len(testData))
    print (testData)
    return labels, trainingData, labelsTest, testData


def innitVars(data, inputNodes, hiddenNodes, outputNodes):
    '''
    Initialise the two weight meshes

    Parameters
    ----------
    inputNodes : int
        The number of nodes in the input layer
    hiddenNodes : int
        The number of nodes in the hidden layer
    outputNodes : int
        The number of nodes in the output layer

    returns
    -------
    inputHiddenConnection : NP.ndarray
        The mesh of weights between the input layer and the hidden layer
    hiddenOutputConnection : NP.ndarray
        The mesh of weights between the hidden layer and the output layer
    '''

    inputHiddenConnection = (NP.random.rand(hiddenNodes, inputNodes) - 0.5)
    hiddenOutputConnection = (NP.random.rand(outputNodes, hiddenNodes) - 0.5)
    return inputHiddenConnection, hiddenOutputConnection


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


def trainNetwork(data, targetArray, learningRate, inputHiddenConnection, hiddenOutputConnection):
    '''
    Function to train the network by calculating errors and updating the weights

    Parameters
    ----------
    data : NP.ndarray
        The training data to be used as inputs
    targetArray : NP.ndarray
        The target output values. Used to calculate error
    learningRate : float
        Hyperparameter used in updating the weights
    inputHiddenConnection : NP.ndarray
        The mesh of weights between the input layer and the hidden layer
    hiddenOutputConnection : NP.ndarray
        The mesh of weights between the hidden layer and the output layer

    returns
    -------
    inputHiddenConnection : NP.ndarray
        The mesh of weights between the input layer and the hidden layer
    hiddenOutputConnection : NP.ndarray
        The mesh of weights between the hidden layer and the output layer
    '''

    inputVals = NP.array(data).transpose()
    targets = NP.array(targetArray).transpose()
    hiddenLayerInputs = NP.dot(inputHiddenConnection, inputVals)
    hiddenLayerOutputs = logisticFunction(hiddenLayerInputs)
    outputLayerInputs = NP.dot(hiddenOutputConnection, hiddenLayerOutputs)
    outputLayerOutputs = logisticFunction(outputLayerInputs)
    outputError = targets - outputLayerOutputs
    hiddenError = NP.dot(hiddenOutputConnection.transpose(), outputError)
    hiddenOutputConnection = hiddenOutputConnection + (learningRate * NP.dot((outputError * outputLayerOutputs * (1 - outputLayerOutputs)), NP.transpose(hiddenLayerOutputs)))
    inputHiddenConnection = inputHiddenConnection + (learningRate * NP.dot((hiddenError * hiddenLayerOutputs * (1 - hiddenLayerOutputs)), NP.transpose(inputVals)))
    return inputHiddenConnection, hiddenOutputConnection


def validation(data, inputHiddenConnection, hiddenOutputConnection, labels):
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

    correctPredsCir, correctPredsPen, correctPredsSqr, correctPredsStr, correctPredsTri = 0, 0, 0, 0, 0

    for i in range(0, len(data)):
        predArray = forwardPropagate(data[i], inputHiddenConnection, hiddenOutputConnection)
        prediction = max(predArray)
        predictionIdx = -1
        
        for j in range(0, len(predArray)):
            predictionIdx += 1
            if predArray[j] == prediction:
                break

        if predictionIdx == labels[i]:
            if labels[i] == 0:
                correctPredsCir += 1
            elif labels[i] == 1:
                correctPredsPen += 1
            elif labels[i] == 2:
                correctPredsSqr += 1
            elif labels[i] == 3:
                correctPredsStr += 1
            else:
                correctPredsTri += 1
        else:
            print (labels[i], predictionIdx)

    return (correctPredsCir/80)*100, (correctPredsPen/80)*100, (correctPredsSqr/80)*100, (correctPredsStr/80)*100, (correctPredsTri/80)*100


def codeRunner():
    '''
    Run from main() -> boilerplate code to manage the algorithm and run functions
    '''

    labels, trainingData, labelsTest, testData = readData('Sheet1')
    scaledTrainingData = (NP.asfarray(trainingData[1:]) / 255.0 * 0.99) + 0.01 # Scale each value to match range of sigmoid function
    scaledTestData = (NP.asfarray(testData[0:]) / 255.0 * 0.99) + 0.01

    numInputNodes, numHiddenNodes, numOutputNodes = 4899, 1000, 5

    #!- Comment out when testing -!#
    for i in range(100): # Range is number of epochs
        print ('Epoch: ', i)
        for j in range(0, len(scaledTrainingData)):
            inputImage = [scaledTrainingData[j]]
            targetArray = [[0.01 for idx in range(5)]]
            #print (int(labels[j]))
            targetArray[0][int(labels[j])] = 0.99
            inputHiddenConnection, hiddenOutputConnection = trainNetwork(inputImage, targetArray, 0.05, inputHiddenConnection, hiddenOutputConnection)
    #!- ************************ -!#

    df = PD.DataFrame(inputHiddenConnection)
    df2 = PD.DataFrame(hiddenOutputConnection)
    df.to_csv('E:/inputHiddenWeights1000_100_0.05.csv', index=False)
    df2.to_csv('E:/hiddenOutputWeights1000_100_0.05.csv', index=False)

    #!- VALIDATION -!#
    inputHiddenConnection = (PD.read_csv('E:/inputHiddenWeights1000_100_0.05.csv')).to_numpy()
    hiddenOutputConnection = (PD.read_csv('E:/hiddenOutputWeights1000_100_0.05.csv')).to_numpy()
    print (len(scaledTestData))
    print (len(labelsTest))
    a1, a2, a3, a4, a5 = validation(scaledTestData, inputHiddenConnection, hiddenOutputConnection, labelsTest)
    print (f'\nValidation Accuracies:\n\nCircle: {a1}%\nPentagon: {a2}%\nSquare: {a3}%\nStar: {a4}%\nTriangle: {a5}%\n')


def main():
    codeRunner()


if __name__ == '__main__':
    main()



