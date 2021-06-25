from Point import Point
from SaliencyMap import SaliencyMap
from WTA import WTA

from numpy import imag
import numpy as np
import cv2
import copy

path = r'localImages\\img.jpg'
image = cv2.imread(path)
#!!! We want only 0.0-1.0 images in the saliency map or things starts to crack
# Never, ever use any other pixel-representation format than 0.0-1.0 float64
image = np.float64(image)/255
# note: Yes, this takes a lot of time. This algorithm's execution time in junkthon is a joke.
saliencyMap = SaliencyMap(image)
# saliencyMap.intensityPyramid.ShowPyramid()
# saliencyMap.colorRPyramid.ShowPyramid()
# saliencyMap.colorGPyramid.ShowPyramid()
# saliencyMap.colorBPyramid.ShowPyramid()
# saliencyMap.colorYPyramid.ShowPyramid()
# saliencyMap.orientation0Pyramid.ShowPyramid()
# saliencyMap.orientation45Pyramid.ShowPyramid()
# saliencyMap.orientation90Pyramid.ShowPyramid()
# saliencyMap.orientation135Pyramid.ShowPyramid()

# todo Here you can show other saliency map steps.

cv2.imshow('Saliency map', saliencyMap.saliencyMap)
cv2.waitKey(0)

# todo when the saliency map is ready please uncomment:
winnerCount = 10  # select how many winners do you want to have, to search for a loop just pass some insanely huge number
wta = WTA(saliencyMap.saliencyMap, blackoutRadius=200, regenerationTime=6)
winners = wta.GetWinnerPoints(6)
winners = wta.WinnerPointsToAnnotationPoints(winners)
annotatedOriginalImage = wta.AnnotateImage(
    image, winners)
cv2.imshow('Annotated original image', annotatedOriginalImage)
cv2.waitKey(0)
annotatedSaliencyMap = wta.Annotate2F1Image(saliencyMap.saliencyMap, winners)
cv2.imshow('Annotated saliency map', annotatedSaliencyMap)
cv2.waitKey(0)

print("Salient points:")
for i in range(0, len(winners)):
    point: Point = winners[i]
    point.Print()
