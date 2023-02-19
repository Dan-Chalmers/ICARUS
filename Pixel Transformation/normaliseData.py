
''' normaliseData.py '''

'''
    !!! TO BE RUN AFTER getPixelValues.py !!!
'''

import pandas as PD
import numpy as NP
import glob
import time


''' Calculate the luminance value of each pixel in each shape in the directory,
    save them too a dataframe and write the dataframes to  '''
def normalisePixelData(directory: str) -> PD.DataFrame:

    completeData = PD.DataFrame(index = NP.arange(1), columns = NP.arange(2170740)) # Initialise new dataframe
    rowAsArray = []
    cntr = 0

    print (completeData)

    for csv in glob.iglob(f'{directory}/*'):
        cnt += 1
        df = PD.read_csv(csv)

        if 'circle' in csv or 'square' in csv:
            numOfRows, numOfCols = 1361, 1401
        elif 'triangle' in csv:
            numOfRows, numOfCols = 1389, 1431
        elif 'pentagon' in csv:
            numOfRows, numOfCols = 1367, 1409
        else:
            numOfRows, numOfCols = 1452, 1495

        for row in range(numOfRows):
            for col in range(numOfCols):
                temp = df.loc[row][col]
                tempList = (temp.strip('[').strip(']').strip(' ')).split(',')
                L = (int(tempList[0]) * 0.299) + (int(tempList[1]) * 0.587) + (int(tempList[2]) * 0.114) # Calculate luminance value of each pixel set with luminance formula
                rowAsArray.append(round(L, 1))
            
        completeData = completeData.append(PD.Series(rowAsArray, index=completeData.columns[:len(rowAsArray)]), ignore_index=True) # Append data to dataframe
        rowAsArray.clear()

    return completeData


''' For testing ONLY - run in isolation'''
def getOneOfEachVal(rowAsArray: list):

    check = []
    for val in rowAsArray:
        if val not in check:
            check.append(val)
    print (check)


def main():
    start = time.time()
    data = normalisePixelData('D://TRAINING_PXL_VALS')
    data.to_csv('D://normalisedData.csv', mode = 'a', index = False, header = False) # Write dataframe to CSV
    end = time.time()
    duration = (end - start) / 60
    print ('\n\nTime: {duration} minutes\n')


if __name__ == '__main__':
    main()