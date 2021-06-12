from SaliencyMap import SaliencyMap
from numpy import imag
import numpy as np
from SaliencyMap import SaliencyMap
import cv2

path = r'images\\i1.jpg'
image = cv2.imread(path)
#!!! We want only 0.0-1.0 images in the saliency map or things starts to crack
# Never, ever use any other pixel-representation format than 0.0-1.0 float64
image = np.float64(image)/255

saliencyMap = SaliencyMap(image)
saliencyMap.intensityPyramid.ShowPyramid()
saliencyMap.colorRPyramid.ShowPyramid()
saliencyMap.colorGPyramid.ShowPyramid()
saliencyMap.colorBPyramid.ShowPyramid()
saliencyMap.colorYPyramid.ShowPyramid()
saliencyMap.orientation0Pyramid.ShowPyramid()
saliencyMap.orientation45Pyramid.ShowPyramid()
saliencyMap.orientation90Pyramid.ShowPyramid()
saliencyMap.orientation135Pyramid.ShowPyramid()

cv2.imshow('Saliency map', saliencyMap.saliencyMap)
cv2.waitKey(0)
