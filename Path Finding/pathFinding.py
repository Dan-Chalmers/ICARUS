
''' pathFinding.py '''

from svgpath2mpl import parse_path as SVG
from svgpathtools import svg2paths

import generateEnvironmentData as GED
import convertToLuminance as C2L
import matplotlib.pyplot as PLT
import compress as CMP
import random as RNDM
import numpy as NP
import cv2 as CV
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append('C:/Users/Dan/Desktop/My Degree/Year 3/6001CEM Individual Project/ANN')
import ANN4sim as Recognise

global path_to_shapes
path_to_shapes = 'C://Users//Dan//Desktop//My Degree//Year 3//6001CEM Individual Project//Shapes//'


def config():

    # Centrepoint -> Heathrow airport runway, radius -> 20 m
    lowerCoordBoundLat, lowercoordBoundLong = 51.47747, -0.45991
    upperCoordBoundLat, upperCoordBoundLong = 51.47787, -0.45951
    return lowerCoordBoundLat, upperCoordBoundLat, lowercoordBoundLong, upperCoordBoundLong


def assignCoordinates(predictions, lbLat, ubLat, lbLon, ubLon):

    coordinateTuples = []

    for i in range(0, len(predictions)):
        randLat = round(RNDM.uniform(lbLat, ubLat), 5)
        randLon = round(RNDM.uniform(lbLon, ubLon), 5)
        coordinateTuples.append((randLat, randLon))
    return coordinateTuples


def plotCoords(coords, predictions):

    circlePath, attCir = svg2paths(path_to_shapes+'circle.svg')
    squarePath, attSqr = svg2paths(path_to_shapes+'square.svg')
    pentagonPath, attPen = svg2paths(path_to_shapes+'pentagon.svg')
    starPath, attStr = svg2paths(path_to_shapes+'star.svg')
    triangle, attTri = svg2paths(path_to_shapes+'triangle.svg')
    cirMarker = SVG(attCir[0]['d'])
    cirMarker.vertices -= cirMarker.vertices.mean(axis=0) # Shift centrepoint of SVG
    sqrMarker = SVG(attSqr[0]['d'])
    sqrMarker.vertices -= sqrMarker.vertices.mean(axis=0)
    penMarker = SVG(attPen[0]['d'])
    penMarker.vertices -= penMarker.vertices.mean(axis=0)
    strMarker = SVG(attStr[0]['d'])
    strMarker.vertices -= strMarker.vertices.mean(axis=0)
    triMarker = SVG(attTri[0]['d'])
    triMarker.vertices -= triMarker.vertices.mean(axis=0)
    PLT.rcParams['figure.figsize'] = [10, 10]
    PLT.rcParams['figure.autolayout'] = True

    for i in range(0, len(coords)):
        if predictions[i] == 0:
            _marker = cirMarker
        elif predictions[i] == 1:
            _marker = sqrMarker
        elif predictions[i] == 2:
            _marker = triMarker
        elif predictions[i] == 3:
            _marker = strMarker
        else:
            _marker = penMarker
        PLT.plot(coords[i][0], coords[i][1], 'o', marker = _marker, markersize = 30, color = 'black')

    PLT.show()


def main():

    dir = 'E:/tempImageData.csv'
    conversions = {0:'circle', 1:'square', 2:'triangle', 3:'star', 4:'pentagon'}
    print ('Configuring environment...')
    lbLat, ubLat, lbLon, ubLon = config()
    files = ['circle', 'square', 'triangle', 'star', 'pentagon']
    for i in range(25):
        idxRandom = RNDM.randint(0, 4)
        img = CV.imread(path_to_shapes+'{0}.png'.format(str(files[idxRandom])))
        img = CV.cvtColor(img, CV.COLOR_RGB2BGR)
        GED.augment(GED.getTransform(), img, i)

    print ('Compression...')
    CMP.compressImage()
    print ('Conversion...')
    C2L.convert('E://tempImageStorage')
    predictions = Recognise.codeRunner(dir)
    print (predictions)
    coords = assignCoordinates(predictions, lbLat, ubLat, lbLon, ubLon)
    print (coords)
    plotCoords(coords, predictions)


if __name__ == '__main__':
    main()