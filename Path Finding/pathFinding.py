
''' pathFinding.py '''

from __future__ import print_function

from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
from svgpath2mpl import parse_path as SVG
from svgpathtools import svg2paths
from pymavlink import mavutil

import generateEnvironmentData as GED
import convertToLuminance as C2L
import matplotlib.pyplot as PLT
import compress as CMP
import random as RNDM
import numpy as NP
import time as T
import math as M
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
    lowerCoordBoundLat, lowercoordBoundLong = 51.47749014, -0.4597999322
    upperCoordBoundLat, upperCoordBoundLong = 51.47784986, -0.4595301357
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
    PLT.rcParams['figure.figsize'] = [7.5, 7.5]
    PLT.rcParams['figure.autolayout'] = True
    PLT.xlabel('Longitude')
    PLT.ylabel('Latitude')
    PLT.title('2D environment abstraction')

    # Keep track of each instance of each shape
    positionsDict = {'circle' : [],
                     'square' : [],
                     'triangle' : [],
                     'star' : [],
                     'pentagon' : []}

    for i in range(0, len(coords)):
        if predictions[i] == 0:
            _marker, _color = cirMarker, 'red'
            positionsDict['circle'].append(coords[i])
        elif predictions[i] == 1:
            _marker, _color = sqrMarker, 'blue'
            positionsDict['square'].append(coords[i])
        elif predictions[i] == 2:
            _marker, _color = triMarker, 'orange'
            positionsDict['triangle'].append(coords[i])
        elif predictions[i] == 3:
            _marker, _color = strMarker, 'green'
            positionsDict['star'].append(coords[i])
        else:
            _marker, _color = penMarker, 'purple'
            positionsDict['pentagon'].append(coords[i])
        PLT.plot(coords[i][1], coords[i][0], 'o', marker = _marker, markersize = 20, color = _color)

    return positionsDict


def shortestPath(positions, shape):
    visited = [(51.47767, -0.45971)]
    unvisited = []
    for i in range(0, len(positions[shape])):
        unvisited.append(positions[shape][i])

    PLT.plot(-0.45971, 51.47767, 'X', markersize = 20, color = 'black')
    
    for i in range(0, len(unvisited)):
        deltas = []
        for coord in unvisited:
            c = M.sqrt((coord[0] - visited[-1][0])**2 + (coord[1] - visited[-1][1])**2)
            deltas.append(c)
        nearestNeighbor = deltas.index(min(deltas))
        PLT.plot([visited[-1][1], unvisited[nearestNeighbor][1]], [visited[-1][0], unvisited[nearestNeighbor][0]], 'k--', linewidth=2)
        visited.append(unvisited[nearestNeighbor])
        unvisited.remove(unvisited[nearestNeighbor])
        print (unvisited)

    PLT.show()
    return visited


def getRemainingDistance(target_location):
    current_location = vehicle.location.global_relative_frame
    dlat = target_location.lat - current_location.lat
    dlong = target_location.lon - current_location.lon
    return M.sqrt((dlat*dlat) + (dlong*dlong))


def simulation(targetAlt, visited):
    sitl = None
    print ('Basic pre-arm checks...')
    while not vehicle.is_armable:
        print ('Waiting for vehicle to initialise...')
        T.sleep(1)

    print ('Arming motors...')
    vehicle.mode = VehicleMode('GUIDED')
    vehicle.armed = True

    while not vehicle.armed:
        print ('Waiting for arming...')
        T.sleep(1)

    # Take off to desired altitude
    print ('Taking off...')
    vehicle.wait_simple_takeoff(targetAlt)

    while True:
        print ('Altitude: ', vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= targetAlt * 0.95:
            print ('Reached target altitude')
            break
        T.sleep(1)

    cmds = vehicle.commands
    vehicle.airspeed = 5
    locationUAVObjects = []
    for i in range(0, len(visited)):
        loc = LocationGlobalRelative(visited[i][0], visited[i][1], 20)
        locationUAVObjects.append(loc)

    for waypoint in locationUAVObjects:
        print ('Flying to shape')
        vehicle.simple_goto(waypoint)
        while True:
            distLeft = getRemainingDistance(waypoint)
            print ('Distance to next shape:', distLeft)
            if distLeft <= 0.000005:
                print ('Shape {shape} reached'.format(shape = locationUAVObjects.index(waypoint)))
                break
            T.sleep(1)

    print ('Route complete')
    vehicle.mode = VehicleMode('LAND')
    while not vehicle.mode.name == 'LAND':
        T.sleep(1)

    vehicle.close()


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
    coords = assignCoordinates(predictions, lbLat, ubLat, lbLon, ubLon)
    positions = plotCoords(coords, predictions)
    visited = shortestPath(positions, 'circle')
    simulation(20, visited)


if __name__ == '__main__':
    vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)
    main()