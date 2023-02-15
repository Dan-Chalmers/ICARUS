
''' getPixelValues.py '''

from PIL import Image as img
import numpy as NP
import glob
import csv


''' Transform the images into numpy arrays, then dataframes, then write to xlsx '''
def translate(directory: str, pxlDirectory: str):

    for image in glob.iglob(f'{directory}/*'):
        shapeArray = NP.asarray(img.open(image))
        processedShape = shapeArray.tolist()

        with open(pxlDirectory+'//{0}'.format(image.strip('.png').strip(directory))+'.csv', 'w', newline='') as pxlVals:
            writer = csv.writer(pxlVals, delimiter=',')
            writer.writerows(processedShape)


def main():

    translate('D://TRAINING', 'D://TRAINING_PXL_VALS')
    translate('D://VALIDATION', 'D://VALIDATION_PXL_VALS')
    

if __name__ == '__main__':
    main()
    