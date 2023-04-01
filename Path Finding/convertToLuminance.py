''' DO NOT COMMIT '''

from PIL import Image as img
import pandas as PD
import numpy as NP
import glob
import os

def convert(directory):

    completeData = PD.DataFrame(index = NP.arange(1), columns = NP.arange(4899)) # Initialise new dataframe
    cnt = 0

    arrayToPush = []

    if os.path.exists('E://tempImageData.csv'):
        os.remove('E://tempImageData.csv')

    for image in glob.iglob(f'{directory}/*'):
        cnt += 1

        shapeArray = NP.asarray(img.open(image).convert('L'))
        processedShape = shapeArray.tolist()
        for i in range(len(processedShape)):
            for j in range(len(processedShape[i])):
                arrayToPush.append(processedShape[i][j])
        completeData = completeData.append(PD.Series(arrayToPush, index=completeData.columns[:len(arrayToPush)]), ignore_index=True)
        completeData.to_csv('E://tempImageData.csv', mode = 'a', index = False, header = False)
        completeData = completeData.iloc[0:0]
        arrayToPush.clear()