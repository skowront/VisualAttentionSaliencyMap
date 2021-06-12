# Implementation based on:
# Itti, Laurent & Koch, Christof & Niebur, Ernst. (1998). A Model of Saliency-based Visual Attention for Rapid Scene Analysis. Pattern Analysis and Machine Intelligence, IEEE Transactions on. 20. 1254 - 1259. 10.1109/34.730558.
# In code citations from the source article are marked with a header line like the one below:
# A Model of Saliency-based Visual Attention for Rapid Scene Analysis:

from typing import List

from numpy.lib import imag
from GaussianPyramid import GaussianPyramid

import numpy as np
import cv2
import copy
import sys


class SaliencyMap:

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Nine spatial scales are created
    # using dyadic Gaussian pyramids [11], which progressively lowpass lter and subsample the input image, yielding horizontal
    # and vertical image reduction factors ranging from 1:1 (scale 0)
    # to 1:256 (scale 8) in eight octaves.
    def __init__(self, image: np.ndarray) -> None:
        self.__image = image
        self.__intensityPyramid = GaussianPyramid(
            image, pyramidHeight=8)
        self.__colorRPyramid = GaussianPyramid(
            image, pyramidHeight=8)
        self.__colorGPyramid = GaussianPyramid(
            image, pyramidHeight=8)
        self.__colorBPyramid = GaussianPyramid(
            image, pyramidHeight=8)
        self.__colorYPyramid = GaussianPyramid(
            image, pyramidHeight=8)
        self.__orientation0Pyramid = GaussianPyramid(
            image, pyramidHeight=1)
        self.__orientation45Pyramid = GaussianPyramid(
            image, pyramidHeight=1)
        self.__orientation90Pyramid = GaussianPyramid(
            image, pyramidHeight=1)
        self.__orientation135Pyramid = GaussianPyramid(
            image, pyramidHeight=1)
        self.__BuildIntensityPyramid()
        self.__maxIntensityOverImage: np.float64 = self.__CalculateMaxIntensity()
        self.__BuildColorRPyramid()
        self.__BuildColorGPyramid()
        self.__BuildColorBPyramid()
        self.__BuildColorYPyramid()
        self.__BuildOrientationPyramids()

    @property
    def intensityPyramid(self):
        return self.__intensityPyramid

    @property
    def colorRPyramid(self):
        return self.__colorRPyramid

    @property
    def colorGPyramid(self):
        return self.__colorGPyramid

    @property
    def colorBPyramid(self):
        return self.__colorBPyramid

    @property
    def colorYPyramid(self):
        return self.__colorYPyramid

    @property
    def Orientation0Pyramid(self):
        return self.__orientation0Pyramid

    @property
    def Orientation45Pyramid(self):
        return self.__orientation45Pyramid

    @property
    def Orientation90Pyramid(self):
        return self.__orientation90Pyramid

    @property
    def Orientation135Pyramid(self):
        return self.__orientation135Pyramid

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # With r, g and b being the red, green and blue channels of the
    # input image, an intensity image I is obtained as I = (r+g+b) = 3.
    # I is used to create a Gaussian pyramid I(), where  2 [0::8]
    # is the scale.
    def __BuildIntensityPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        intensityImg = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                intensityImg[i][j] = np.float64(
                    pixel[0]+pixel[1]+pixel[2])/3
        self.__intensityPyramid.original = intensityImg

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # However, because hue
    # variations are not perceivable at very low luminance (and hence
    # are not salient), normalization is only applied at the locations
    # where I is larger than 1=10 of its maximum over the entire
    # image (other locations yield zero r; g and b).
    def __CalculateMaxIntensity(self):
        max = self.__intensityPyramid.original[0][0]
        for i in range(0, len(self.__intensityPyramid.original)):
            for j in range(0, len(self.__intensityPyramid.original[i])):
                if self.__intensityPyramid.original[i][j] > max:
                    max = self.__intensityPyramid.original[i][j]
        return max

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # R = r  (g + b)=2 for red
    def __BuildColorRPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1*self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1/self.intensityPyramid.original[i][j])
        # channel separation
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                r = np.float64(pixel[0])
                g = np.float64(pixel[1])
                b = np.float64(pixel[2])
                img[i][j] = np.float64(r-((g+b)/2))
        self.__colorRPyramid.original = img

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # G = g(r + b)
    def __BuildColorGPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1*self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1/self.intensityPyramid.original[i][j])
        # channel separation
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                r = np.float64(pixel[0])
                g = np.float64(pixel[1])
                b = np.float64(pixel[2])
                img[i][j] = np.float64(g-((r+b)/2))
        self.__colorGPyramid.original = img

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # B = b  (r + g)
    def __BuildColorBPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1*self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1/self.intensityPyramid.original[i][j])
        # channel separation
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                r = np.float64(pixel[0])
                g = np.float64(pixel[1])
                b = np.float64(pixel[2])
                img[i][j] = np.float64(b-((r+g)/2))
        self.__colorBPyramid.original = img

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Y = (r + g)/2 - |r - g|/2 - b
    def __BuildColorYPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1*self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1/self.intensityPyramid.original[i][j])
        # channel separation
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                r = np.float64(pixel[0])
                g = np.float64(pixel[1])
                b = np.float64(pixel[2])
                img[i][j] = np.float64(((r+g)/2)-(abs(r-g)/2 - b))
        self.__colorYPyramid.original = img

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Local orientation information is obtained from I using
    # oriented Gabor pyramids O(fi,theta) where theta is in [0...8]
    # represents the scale and theta is in {0deg,45deg,90deg,135deg}
    # is the preferred orientation
    def __BuildOrientationPyramids(self):
        img = self.intensityPyramid.original
        self.__orientation0Pyramid.original = self.__BuildGaborImage(
            img=img, theta=0)
        self.__orientation45Pyramid.original = self.__BuildGaborImage(
            img=img, theta=45)
        self.__orientation90Pyramid.original = self.__BuildGaborImage(
            img=img, theta=90)
        self.__orientation135Pyramid.original = self.__BuildGaborImage(
            img=img, theta=135)
        return

    # Just a helper function for gabor images, use this instead of __BuildGaborFilterImage()
    def __BuildGaborImage(self, img, theta):
        filters = self.__BuildGaborFilters(theta)
        return self.__BuildGaborFilterImage(img, filters)

    # Dont use this, implements
    # https: // cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practic
    def __BuildGaborFilters(self, theta):
        filters = []
        ksize = 31
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta,
                                  10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters

    # Dont use this, implements
    # https: // cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
    def __BuildGaborFilterImage(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum
