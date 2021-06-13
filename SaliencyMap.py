# Implementation based on:
# Itti, Laurent & Koch, Christof & Niebur, Ernst. (1998). A Model of Saliency-based Visual Attention for Rapid Scene Analysis. Pattern Analysis and Machine Intelligence, IEEE Transactions on. 20. 1254 - 1259. 10.1109/34.730558.
# In code citations from the source article are marked with a header line like the one below:
# A Model of Saliency-based Visual Attention for Rapid Scene Analysis:

from typing import List, no_type_check_decorator
from numpy.core.fromnumeric import shape

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
        self.__intensityPyramid = self.__BuildIntensityPyramid(8)
        self.__maxIntensityOverImage: np.float64 = self.__CalculateMaxIntensity()
        self.__colorRPyramid = self.__BuildColorRPyramid(8)
        self.__colorGPyramid = self.__BuildColorGPyramid(8)
        self.__colorBPyramid = self.__BuildColorBPyramid(8)
        self.__colorYPyramid = self.__BuildColorYPyramid(8)
        self.__orientation0Pyramid = self.__BuildOrientationPyramid(8, 0)
        self.__orientation45Pyramid = self.__BuildOrientationPyramid(8, 45)
        self.__orientation90Pyramid = self.__BuildOrientationPyramid(8, 90)
        self.__orientation135Pyramid = self.__BuildOrientationPyramid(8, 135)
        self.__BuildIntensityFeatureMaps()
        self.__BuildColorFeatureMaps()
        self.__BuildOrientationFeatureMaps()
        self.__conspicuityIntensityMap = self.__BuildConspicuityInensityMap()
        self.__conspicuityColorMap = self.__BuildConspicuityColorMap()
        self.__conspicuityOrientationMap = self.__BuildConspicuityOrientationMap()
        self.__saliencyMap = self.__BuildSaliencyMap()

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
    def orientation0Pyramid(self):
        return self.__orientation0Pyramid

    @property
    def orientation45Pyramid(self):
        return self.__orientation45Pyramid

    @property
    def orientation90Pyramid(self):
        return self.__orientation90Pyramid

    @property
    def orientation135Pyramid(self):
        return self.__orientation135Pyramid

    @property
    def saliencyMap(self):
        return self.__saliencyMap

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # With r, g and b being the red, green and blue channels of the
    # input image, an intensity image I is obtained as I = (r+g+b) = 3.
    # I is used to create a Gaussian pyramid I(), where  2 [0::8]
    # is the scale.
    def __BuildIntensityPyramid(self, height: int) -> GaussianPyramid:
        srcImg = copy.deepcopy(self.__image)
        intensityImg = np.zeros(
            shape=(srcImg.shape[0], srcImg.shape[1]), dtype=np.float64)
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                intensityImg[i][j] = np.float64(
                    pixel[0]+pixel[1]+pixel[2])/3
        return GaussianPyramid(intensityImg, height)

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
    def __BuildColorRPyramid(self, height: int) -> GaussianPyramid:
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
        return GaussianPyramid(img, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # G = g(r + b)
    def __BuildColorGPyramid(self, height: int) -> GaussianPyramid:
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
        return GaussianPyramid(img, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # B = b  (r + g)
    def __BuildColorBPyramid(self, height: int) -> GaussianPyramid:
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
        return GaussianPyramid(img, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Y = (r + g)/2 - |r - g|/2 - b
    def __BuildColorYPyramid(self, height: int) -> GaussianPyramid:
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
                img[i][j] = np.float64(((r+g)/2)-(abs(r-g)/2) - b)
        return GaussianPyramid(img, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Local orientation information is obtained from I using
    # oriented Gabor pyramids O(fi,theta) where theta is in [0...8]
    # represents the scale and theta is in {0deg,45deg,90deg,135deg}
    # is the preferred orientation
    def __BuildOrientationPyramid(self, height, theta) -> GaussianPyramid:
        img = self.intensityPyramid.original
        gbImage = self.__BuildGaborImage(img=img, theta=theta)
        return GaussianPyramid(gbImage, height)

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
            fimg = cv2.filter2D(img, cv2.CV_64FC1, kern)
            np.maximum(accum, fimg, accum)
        return accum

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Center-surround dierences (	 dened previously) between a
    # \center" ne scale c and a \surround" coarser scale s yield the
    # feature maps. The rst set of feature maps is concerned with
    # intensity contrast, which in mammals is detected by neurons
    # sensitive either to dark centers on bright surrounds, or to bright
    # centers on dark surrounds [12]. Here, both types of sensitivities
    # are simultaneously computed (using a rectication) in a set of
    # six maps I(c; s), with c 2 f2; 3; 4g and s = c + ,  2 f3; 4g:
    # I(c; s) = jI (c) 	 I
    def __BuildIntensityFeatureMaps(self):
        # todo
        pass

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # A second set of maps is similarly constructed for the color
    # channels, which in cortex are represented using a so-called \color
    # double-opponent" system: In the center of their receptive eld,
    # neurons are excited by one color (e.g., red) and inhibited by
    # another (e.g., green), while the converse is true in the surround.
    # Such spatial and chromatic opponency exists for the red/green,
    # green/red, blue/yellow and yellow/blue color pairs in human
    # primary visual cortex [13]. Accordingly, maps RG(c; s) are
    # created in the model to simultaneously account for red/green
    # and green/red double opponency (Eq. 2), and BY(c; s) for
    # blue/yellow and yellow/blue double opponency (Eq. 3):
    # RG(c; s) = j(R(c)  G(c)) 	 (G(s)  R(s))j (2)
    # BY(c; s) = j(B(c)  Y (c)) 	 (Y (s)
    def __BuildColorFeatureMaps(self):
        # todo
        pass

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Orientation
    # feature maps, O(c; s; ), encode, as a group, local orientation
    # contrast between the center and surround scales:
    # O(c; s; )=jO(c; )
    def __BuildOrientationFeatureMaps(self):
        # todo
        pass

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Center-surround is implemented in the model as the dierence between ne and
    # coarse scales: The center is a pixel at scale c {2; 3; 4} ,and
    # the surround is the corresponding pixel at scale s = c + delta, with
    # delta {3; 4}
    def __CenterSurroundOperator(self):
        # todo
        pass

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # In the absence of top-down supervision, we propose a map
    # normalization operator, N (:), which globally promotes maps
    # in which a small number of strong peaks of activity (conspicuous locations) is present, while globally suppressing maps which
    # contain numerous comparable peak responses. N (:) consists of
    # (Fig. 2): 1) Normalizing the values in the map to a xed range
    # [0::M], in order to eliminate modality-dependent amplitude differences; 2) nding the location of the map's global maximum
    # M and computing the average m of all its other local maxima;
    # 3) globally multiplying the map by (M-m)^2.
    def __NormalizeOperator(self, image: np.ndarray) -> np.ndarray:
        # todo
        # remove line below after todo implemetnat
        return image

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (5)
    def __BuildConspicuityInensityMap(self) -> np.ndarray:
        # todo
        # remove line below after todo implemetnation
        return np.zeros(self.__image.shape[:2])

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (6)
    def __BuildConspicuityColorMap(self) -> np.ndarray:
        # todo
        # remove line below after todo implemetnation
        return np.zeros(self.__image.shape[:2])

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (7)

    def __BuildConspicuityOrientationMap(self) -> np.ndarray:
        # todo
        # remove line below after todo implemetnation
        return np.zeros(self.__image.shape[:2])

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (8)
    def __BuildSaliencyMap(self) -> np.ndarray:
        iDash = self.__conspicuityIntensityMap
        cDash = self.__conspicuityColorMap
        oDash = self.__conspicuityOrientationMap
        nIDash = self.__NormalizeOperator(iDash)
        nCDash = self.__NormalizeOperator(cDash)
        nODash = self.__NormalizeOperator(oDash)
        s = np.zeros(nIDash.shape[:2])
        for i in range(0, len(nIDash)):
            for j in range(0, len(nIDash[i])):
                s[i][j] = (nIDash[i][j]+nCDash[i][j]+nODash[i][j])/3
        return s
