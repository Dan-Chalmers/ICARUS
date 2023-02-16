
''' getPixelValues.py '''

from PIL import Image as img
import numpy as NP
import glob
import csv
import os


''' Transform the images into numpy arrays, then dataframes, then write to xlsx '''
def translate(directory: str, pxlDirectory: str):

    for image in glob.iglob(f'{directory}/*'):
        
        pathToCSV = pxlDirectory+'//{0}'.format(image.strip('.png').strip(directory))+'.csv'
        if not os.path.exists(pathToCSV):
            shapeArray = NP.asarray(img.open(image))
            processedShape = shapeArray.tolist()

            with open(pathToCSV, 'w', newline='') as pxlVals:
                writer = csv.writer(pxlVals, delimiter=',')
                writer.writerows(processedShape)

        else:
            print ('void')


def main():

    translate('D://TRAINING', 'D://TRAINING_PXL_VALS')
    translate('D://VALIDATION', 'D://VALIDATION_PXL_VALS')
    

if __name__ == '__main__':
    main()
    