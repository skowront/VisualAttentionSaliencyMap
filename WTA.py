from typing import List

from numpy.core.records import fromrecords
from numpy.lib import imag
from numpy.lib.type_check import _imag_dispatcher
from Neuron import Neuron
from Point import Point
from Neuron import Neuron
import numpy as np
import cv2
import copy

# Da ting go skkkkrrrraaa, papakakaka
# Skivipipopop and a poopooturrrboom
# Skrra, tutukukututoom, poompoom


class WTA:
    def __init__(self, wbImage: np.ndarray, blackoutRadius: int, regenerationTime: int) -> None:
        self.__image = wbImage
        self.__regenerationTime = regenerationTime
        self.__blackoutRadius = blackoutRadius
        self.__neuronArray = self.__NeuronArrayFromImage(self.__image)
        self.__currentWinner = Point()

    @property
    def image(self):
        return self.__image

    # Returns a 2D ndarray that represents current state of the WTA.
    @property
    def currentImage(self) -> np.ndarray:
        image = np.zeros(self.__neuronArray.shape, dtype=float)
        for i in range(0, len(self.__neuronArray)):
            for j in range(0, len(self.__neuronArray[i])):
                neuron: Neuron = self.__neuronArray[i][j]
                image[i][j] = neuron.CurrentValue
        return image

    # Resets the tournament state to inital.
    def Reset(self):
        self.__neuronArray = self.__NeuronArrayFromImage(self.__image)
        self.__currentWinner = Point()

    # Converts a wb 2F1 image to a neuron array
    def __NeuronArrayFromImage(self, wbImage: np.ndarray) -> np.ndarray:
        neuronArray = np.zeros((wbImage.shape), dtype=Neuron)
        for i in range(0, len(neuronArray)):
            for j in range(0, len(neuronArray[i])):
                neuron = Neuron(
                    0, wbImage[i][j], self.__regenerationTime)
                neuron.Fill()
                neuronArray[i][j] = neuron
        return neuronArray

    # Goes to the next winner.
    def Next(self):
        self.__currentWinner = self.__SelectWinner()
        self.__BlackOutArea(self.__currentWinner, self.__blackoutRadius)
        for i in range(0, len(self.__neuronArray)):
            for j in range(0, len(self.__neuronArray[i])):
                neuron: Neuron = self.__neuronArray[i][j]
                neuron.Regenerate()

    # Blackouts an area with given center point and radius
    def __BlackOutArea(self, centerPoint: Point, radius: int):
        for i in range(0, len(self.__neuronArray)):
            for j in range(0, len(self.__neuronArray[i])):
                xDist = abs(centerPoint.X-i)
                yDist = abs(centerPoint.Y-j)
                dist = np.sqrt((xDist*xDist)+(yDist*yDist))
                if dist <= radius:
                    neuron: Neuron = self.__neuronArray[i][j]
                    neuron.Zero()

    # Selects a winner.
    def __SelectWinner(self) -> Point:
        cImage = self.currentImage
        max = cImage[0][0]
        maxCoords = Point(0, 0)
        for i in range(0, len(cImage)):
            for j in range(0, len(cImage[i])):
                if cImage[i][j] > max:
                    max = cImage[i][j]
                    maxCoords = Point(i, j)
        return maxCoords

    # Gets a given number of winners
    def GetWinnerPoints(self, count: int) -> List[Point]:
        self.Reset()
        winners = list()
        for i in range(0, count):
            self.Next()
            winner = self.__currentWinner
            winners.append(winner)
        return winners

    # Displays current state of the tournament.
    def ShowCurrent(self):
        cv2.imshow('Saliency map', self.currentImage)
        cv2.waitKey(0)

    # image must be a 2D array with float values.
    # if you want to annotate an RGB image please use AnnotateImage
    def Annotate2F1Image(self, image, winnerPoints, annotateCircles: bool = True, annotateArrows: bool = True, annotateIndexes: bool = True) -> np.ndarray:
        cImage = image
        img = np.zeros(
            (cImage.shape[0], cImage.shape[1], 3), np.float64)
        rad = int(cImage.shape[0]/20)
        if rad < 1:
            rad = 1
        thick = int(cImage.shape[0]/150)
        if thick < 1:
            thick = 1
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                img[i][j][0] = cImage[i][j]
                img[i][j][1] = cImage[i][j]
                img[i][j][2] = cImage[i][j]
        if annotateArrows == True:
            for i in range(0, len(winnerPoints)):
                winner: Point = winnerPoints[i]
                if i < len(winnerPoints)-1:
                    nextWinner: Point = winnerPoints[i+1]
                    cv2.arrowedLine(img=img, pt1=(winner.X, winner.Y),
                                    pt2=(nextWinner.X, nextWinner.Y), color=[1.0, 0.0, 0.0], thickness=int(thick*0.8), tipLength=0.03, line_type=cv2.LINE_AA)
        if annotateCircles == True:
            for i in range(0, len(winnerPoints)):
                winner: Point = winnerPoints[i]
                cv2.circle(img=img, center=(winner.X, winner.Y),
                           radius=rad, color=[0, 1.0, 1.0], thickness=thick, lineType=cv2.LINE_AA)
        if annotateIndexes:
            for i in range(0, len(winnerPoints)):
                winner: Point = winnerPoints[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img, text=str(i), org=(
                    winner.X, winner.Y), fontFace=font, fontScale=0.5, color=[0.0, 0.0, 1.0])
        return img

    def WinnerPointsToAnnotationPoints(self, winnerPoints) -> List:
        annotations = []
        for i in range(0, len(winnerPoints)):
            pt: Point = winnerPoints[i]
            annotations.append(Point(pt.Y, pt.X))
        return annotations
        # Annotates any image with given winner points.

    def AnnotateImage(self, image, winnerPoints, annotateCircles: bool = True, annotateArrows: bool = True, annotateIndexes: bool = True) -> np.ndarray:
        img = copy.deepcopy(image)
        rad = int(image.shape[0]/20)
        if rad < 1:
            rad = 1
        thick = int(image.shape[0]/150)
        if thick < 1:
            thick = 1
        if annotateArrows == True:
            for i in range(0, len(winnerPoints)):
                winner: Point = winnerPoints[i]
                if i < len(winnerPoints)-1:
                    nextWinner: Point = winnerPoints[i+1]
                    cv2.arrowedLine(img=img, pt1=(winner.X, winner.Y),
                                    pt2=(nextWinner.X, nextWinner.Y), color=[1.0, 0.0, 0.0], thickness=int(thick*0.8), tipLength=0.03, line_type=cv2.LINE_AA)
        if annotateCircles == True:
            for i in range(0, len(winnerPoints)):
                winner: Point = winnerPoints[i]
                cv2.circle(img=img, center=(winner.X, winner.Y),
                           radius=rad, color=[0, 1.0, 1.0], thickness=thick, lineType=cv2.LINE_AA)
        if annotateIndexes:
            for i in range(0, len(winnerPoints)):
                winner: Point = winnerPoints[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img, text=str(i), org=(
                    winner.X, winner.Y), fontFace=font, fontScale=0.5, color=[0.0, 0.0, 1.0])
        return img

    # Test function for WTA.
    def Test():
        test = np.zeros(shape=(500, 500))

        test[0][0] = 10.0
        test[9][0] = 15.0
        test[50][50] = 75.0
        test[150][100] = 300.0
        test[4][3] = 10.0
        test[6][4] = 10.0
        test[300][450] = 50.0
        test[250][100] = 100.0

        wta = WTA(test, 2)
        winners = wta.GetWinnerPoints(3)
        for item in winners:
            print(item)

        img = wta.Annotate2F1Image(wta.image, winners)
        cv2.imshow('Saliency map', img)
        cv2.waitKey(0)

# Uncomment to test WTA.
# WTA.Test()
