from SaliencyMap import SaliencyMap
from numpy import imag
import numpy as np
from SaliencyMap import SaliencyMap
import cv2

path = r'images\\i1.jpg'
image = cv2.imread(path)
image = np.float64(image)/255

saliencyMap = SaliencyMap(image)
saliencyMap.intensityPyramid.ShowPyramid()
saliencyMap.colorRPyramid.ShowPyramid()
saliencyMap.colorGPyramid.ShowPyramid()
saliencyMap.colorBPyramid.ShowPyramid()
saliencyMap.colorYPyramid.ShowPyramid()
