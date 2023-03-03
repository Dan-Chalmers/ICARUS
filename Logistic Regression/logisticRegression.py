
''' logisticRegression.py '''


import numpy as NP
import pandas as PD
import seaborn as SBN
import dask.dataframe as Ddf
import matplotlib.pyplot as plt

from itertools import chain as CHN
from sklearn.metrics import confusion_matrix as conf


def logisticFunction(x):
    '''
    Perform a sigmoid calculation on the parameter

    parameters
    ----------
    x : float
        The multiplication of weights.T & trainingData with the bias value added

    returns
    -------
    a : float
        Result of the sigmoid calculation
    '''

    a = 1 / (1 + NP.exp(-x))
    return a


def innitVars(numOfImages):
    '''
    Set weights & bias to an array of random values & 0 respectively

    prameters
    ---------
    numOfImages : int
        An integer equal to the number of images in the dataset

    returns
    -------
    weights : NP.ndarray
        The randomised weights
    bias : int
        The initial bias, set to 0
    '''

    weights = NP.zeros((numOfImages, 1))
    bias = 0
    return weights, bias


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
    trainingData = Ddf.read_csv('E://compressed5000PixelData.csv') # read in CSV in parallel uising dask dataframe
    trainingData = trainingData.compute() # Join all partitioned dask dataframes in a pandas df
    trainingData = trainingData.to_numpy()
    testData = PD.read_csv('E://compressed5000ValidationPixelData.csv')
    testData = testData.to_numpy()
    return labels, trainingData, labelsTest, testData


def regress(weights, bias, labels, trainingData):
    '''
    Forward propagate through the training set, calculating the predicted value and cost fucntion and update the weights and bias

    parameters
    ----------
    weights : NP.ndarray
        An array of the weight values to be updated
    bias : float
        The current bias value (0 if on first iteration)
    labels : NP.ndarray
        An array of the image labels. Size (1, numPixels)
    trainingData : NP.ndarray
        An array containing the images' pixel data. Size (numPixels, numImages)

    returns
    -------
    gradients : dict
        A dictionary holding the derivatives (gradient) of the weights and bias
    costFunction : float
        The calculated cost function
    '''

    datasetSize = trainingData.shape[1] # Number of images
    predictedValue = logisticFunction(NP.dot(weights.transpose(), trainingData) + bias) # Prediction
    costFunction = (-1 / datasetSize) * NP.sum(labels * NP.log(predictedValue) + (1 - labels) * NP.log(1 - predictedValue)) # Calculate Cost funtion
    gradientWeights = (1 / datasetSize) * NP.dot(trainingData, (predictedValue - labels).transpose())
    gradientBias = (1 / datasetSize) * NP.sum(predictedValue - labels)
    gradients = {'dW': gradientWeights, 'dB': gradientBias}
    return gradients, costFunction


def optimiseParams(weights, bias, labels, trainingData, numIterations, learningRate):
    '''
    Run the 'regress' function multiple times to optimise the parameters and return the optimal values

    parameters
    ----------
    weights: NP.ndarray
        An array of the weight values to be updated
    bias : float
        The current bias value (0 if on first iteration)
    labels : NP.ndarray
        An array of the image labels. Size (1, numPixels)
    trainingData : NP.ndarray
        An array containing the images' pixel data. Size (numPixels, numImages)
    numIterations : int
        An integer equal to the value of how may iterations the function should call 'regress'
    learningRate : float
        A float equial to the value of the learning rate

    returns
    -------
    params : dict
        Contains the final values for weights and bias
    gradients : dict
        Contains the final values for the derivatives of weights and bias
    costFuncs : list
        A list of some of the cost fucntions
    '''

    costFuncs = []
    for i in range(numIterations):
        g, c = regress(weights, bias, labels, trainingData)
        gradientWeights, gradientBias = g['dW'], g['dB']
        weights = weights - learningRate * gradientWeights
        bias = bias - learningRate * gradientBias
        if i % 20 == 0:
            costFuncs.append(c)

    params = {'weights': weights, 'bias': bias}
    gradients = {'gradientWeights': gradientWeights, 'gradientBias': gradientBias}
    return params, gradients, costFuncs


def makePredict(weights, bias, trainingData):

    oneDPrediction = []
    valuePrediction = logisticFunction(NP.dot(weights.T, trainingData) + bias)
    prediction = NP.zeros((1, trainingData.shape[1]))
    for i in range(valuePrediction.shape[1]):
        if valuePrediction[0,i] > 0.5:
            prediction[0,i] = 1
        else:
            prediction[0,i] = 0

    for elem in prediction[0]:
        oneDPrediction.append(int(elem))

    return oneDPrediction


def validation(labelsTest, pred, shape):

    correct = 0
    for i in range(0, len(labelsTest)):
        print (labelsTest[i], pred[i])
        if labelsTest[i] == pred[i]:
            correct += 1
        else:
            print ('HAZAAAAR')

    accuracy = (correct / len(labelsTest)) * 100
    confMatrix = conf(labelsTest, pred)
    axis = plt.subplot()
    SBN.heatmap(confMatrix, annot=True, fmt='g', ax=axis)
    axis.set_xlabel('Predicted labels')
    axis.set_ylabel('Actual labels')
    axis.set_title('{shape} Confusion Matrix'.format(shape=shape))
    axis.xaxis.set_ticklabels(['0', '1'])
    axis.yaxis.set_ticklabels(['0', '1'])
    plt.show()
    return accuracy

def main():
    
    shape = 'Triangle'
    labels, trainingData, labelsTest, testData = readData(shape.lower())
    labels, trainingData, testData = labels.transpose(), trainingData.transpose(), testData.T # Inverse shape

    accs = []

    weights, bias = innitVars(trainingData.shape[0])
    
    params, gradients, costFuncs = optimiseParams(weights, bias, labels, trainingData, 1000, 0.01)
    w = params['weights']
    b = params['bias']
    print (params)
    print ('\n')
    print (gradients)
    print ('\n')
    print (costFuncs)
    pred = makePredict(w, b, testData)
    print (pred)
    print (labelsTest)
    acc = validation(labelsTest, pred, shape)
    print ('Accuracy =', acc)
    accs.append(acc)
    



    #print (d)


if __name__ == '__main__':
    main()