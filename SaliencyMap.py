from typing import List
from GaussianPyramid import GaussianPyramid

import numpy as np
import cv2
import copy


class SaliencyMap:
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
        self.__orientationPyramid = GaussianPyramid(
            image, pyramidHeight=1)
        self.__BuildIntensityPyramid()
        self.__maxIntensityOverImage: np.float64 = self.__CalculateMaxIntensity()
        self.__BuildColorRPyramid()
        self.__BuildColorGPyramid()
        self.__BuildColorBPyramid()
        self.__BuildColorYPyramid()

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
    def colorOrientationPyramid(self):
        return self.__orientationPyramid

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

    def __CalculateMaxIntensity(self):
        max = self.__intensityPyramid.original[0][0]
        for i in range(0, len(self.__intensityPyramid.original)):
            for j in range(0, len(self.__intensityPyramid.original[i])):
                if self.__intensityPyramid.original[i][j] > max:
                    max = self.__intensityPyramid.original[i][j]
        return max

    def __BuildColorRPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
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

    def __BuildColorGPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
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

    def __BuildColorBPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
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

    def __BuildColorYPyramid(self):
        srcImg = copy.deepcopy(self.__image)
        img = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
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
                img[i][j] = np.float64(((r+g)/2)-(abs(r-g)/2))
        self.__colorYPyramid.original = img
