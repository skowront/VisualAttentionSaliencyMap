# Implementation based on:
# Itti, Laurent & Koch, Christof & Niebur, Ernst. (1998). A Model of Saliency-based Visual Attention for Rapid Scene Analysis. Pattern Analysis and Machine Intelligence, IEEE Transactions on. 20. 1254 - 1259. 10.1109/34.730558.
# In code citations from the source article are marked with a header line like the one below:
# A Model of Saliency-based Visual Attention for Rapid Scene Analysis:

from typing import List, no_type_check_decorator
from numpy.core.fromnumeric import shape
from numpy.core.numeric import outer

from numpy.lib import imag
from GaussianPyramid import GaussianPyramid

import numpy as np
import cv2
import copy
import sys

from progressBar import printProgressBar


class SaliencyMap:

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Nine spatial scales are created
    # using dyadic Gaussian pyramids [11], which progressively lowpass lter and subsample the input image, yielding horizontal
    # and vertical image reduction factors ranging from 1:1 (scale 0)
    # to 1:256 (scale 8) in eight octaves.
    def __init__(self, image: np.ndarray) -> None:
        progressLen = 14
        printProgressBar(0, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__image = image
        self.__centers = [2, 3, 4]
        self.__deltas = [3, 4]
        self.__angles = [0, 45, 90, 135]
        self.__intensityPyramid = self.__BuildIntensityPyramid(8)
        self.__maxIntensityOverImage: np.float64 = self.__CalculateMaxIntensity()
        self.__colorRPyramid = self.__BuildColorRPyramid(8)
        print("RPyramid")
        printProgressBar(1, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__colorGPyramid = self.__BuildColorGPyramid(8)
        print("GPyramid")
        printProgressBar(2, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__colorBPyramid = self.__BuildColorBPyramid(8)
        print("BPyramid")
        printProgressBar(3, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__colorYPyramid = self.__BuildColorYPyramid(8)
        print("YPyramid")
        printProgressBar(4, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__orientation0Pyramid = self.__BuildOrientationPyramid(8, 0)
        print("O0Pyramid")
        printProgressBar(5, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__orientation45Pyramid = self.__BuildOrientationPyramid(8, 45)
        print("O45Pyramid")
        printProgressBar(6, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__orientation90Pyramid = self.__BuildOrientationPyramid(8, 90)
        print("O90Pyramid")
        printProgressBar(7, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__orientation135Pyramid = self.__BuildOrientationPyramid(8, 135)
        print("O135Pyramid")
        printProgressBar(8, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__intensityFeatureMaps = self.__BuildIntensityFeatureMaps()
        print("IFeature")
        printProgressBar(9, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__colorFeatureMaps = self.__BuildColorFeatureMaps()
        print("CFeature")
        printProgressBar(10, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__orientationFeatureMaps = self.__BuildOrientationFeatureMaps()
        print("OFeature")
        printProgressBar(11, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__conspicuityIntensityMap = self.__BuildConspicuityInensityMap()
        print("IDashPyramid")
        printProgressBar(12, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__conspicuityColorMap = self.__BuildConspicuityColorMap()
        print("CDashPyramid")
        printProgressBar(13, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
        self.__conspicuityOrientationMap = self.__BuildConspicuityOrientationMap()
        print("ODashPyramid")
        printProgressBar(14, progressLen, prefix='Progress:',
                         suffix='Complete', length=50)
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
    def intensityFeatureMap(self):
        return self.__intensityFeatureMaps

    @property
    def colorFeatureMap(self):
        return self.__colorFeatureMaps

    @property
    def orientationFeatureMaps(self):
        return self.__orientationFeatureMaps

    @property
    def conspicuityIntensityMap(self):
        return self.__conspicuityIntensityMap

    @property
    def conspicutyColorMap(self):
        return self.__conspicuityColorMap

    @property
    def conspicutyOrientationMap(self):
        return self.__conspicuityOrientationMap

    @property
    def saliencyMap(self):
        return self.__saliencyMap

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # With r, g and b being the red, green and blue channels of the
    # input image, an intensity image I is obtained as I = (r+g+b) = 3.
    # I is used to create a Gaussian pyramid I(), where  2 [0::8]
    # is the scale.

    def __BuildIntensityPyramid(self, height: int) -> GaussianPyramid:
        r, g, b = self.__image[:, :, 0], self.__image[:,
                                                      :, 1], self.__image[:, :, 2]
        intensityImg = np.float64(r + g + b) / 3
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
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1 * self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1 / self.intensityPyramid.original[i][j])
        # channel separation
        r, g, b = srcImg[:, :, 0], srcImg[:, :, 1], srcImg[:, :, 2]
        R = np.float64(r - ((g + b) / 2))
        return GaussianPyramid(R, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # G = g(r + b)
    def __BuildColorGPyramid(self, height: int) -> GaussianPyramid:
        srcImg = copy.deepcopy(self.__image)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1 * self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1 / self.intensityPyramid.original[i][j])
        # channel separation
        r, g, b = srcImg[:, :, 0], srcImg[:, :, 1], srcImg[:, :, 2]
        G = np.float64(g - ((r + b) / 2))
        return GaussianPyramid(G, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # B = b  (r + g)
    def __BuildColorBPyramid(self, height: int) -> GaussianPyramid:
        srcImg = copy.deepcopy(self.__image)
        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1 * self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1 / self.intensityPyramid.original[i][j])
        # channel separation
        r, g, b = srcImg[:, :, 0], srcImg[:, :, 1], srcImg[:, :, 2]
        B = np.float64(b - ((r + g) / 2))
        return GaussianPyramid(B, height)

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Y = (r + g)/2 - |r - g|/2 - b
    def __BuildColorYPyramid(self, height: int) -> GaussianPyramid:
        srcImg = copy.deepcopy(self.__image)

        # normalization, i guess. If this becomes a problem just kick this double for loop out
        # Reason why we do this is marked above __CalculateMaxIntensity()
        for i in range(0, len(srcImg)):
            for j in range(0, len(srcImg[i])):
                pixel = srcImg[i][j]
                if self.__intensityPyramid.original[i][j] > 0.1 * self.__maxIntensityOverImage:
                    srcImg[i][j] = srcImg[i][j] * \
                        (1 / self.intensityPyramid.original[i][j])
        # channel separation
        r, g, b = srcImg[:, :, 0], srcImg[:, :, 1], srcImg[:, :, 2]
        Y = np.float64(((r + g) / 2) - (abs(r - g) / 2) - b)

        for i, ySub in enumerate(Y):
            for j, y in enumerate(ySub):
                if y < 0:
                    Y[i][j] = 0

        return GaussianPyramid(Y, height)

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
        kern /= 1.5 * kern.sum()
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
    def __BuildIntensityFeatureMaps(self) -> list:
        intensityFeatureMaps = []

        for delta in self.__deltas:
            for center in self.__centers:
                pyramidCenter = self.__BuildIntensityPyramid(center)
                pyramidScale = self.__BuildIntensityPyramid(center + delta)
                centerOp = self.__CenterSurroundOperatorAbsolute(
                    pyramidCenter, pyramidScale)
                for item in centerOp:
                    intensityFeatureMaps.append(item)

        return intensityFeatureMaps

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
    def __BuildColorFeatureMaps(self) -> (list, list):
        RG = []
        BY = []

        for delta in self.__deltas:
            for center in self.__centers:
                pyramidCenterR = self.__BuildColorRPyramid(center)
                pyramidCenterG = self.__BuildColorGPyramid(center)
                pyramidCenterB = self.__BuildColorBPyramid(center)
                pyramidCenterY = self.__BuildColorYPyramid(center)
                pyramidScaleR = self.__BuildColorRPyramid(center + delta)
                pyramidScaleG = self.__BuildColorGPyramid(center + delta)
                pyramidScaleB = self.__BuildColorBPyramid(center + delta)
                pyramidScaleY = self.__BuildColorYPyramid(center + delta)
                centerOp = (self.__CenterSurroundOperatorAbsolute(pyramidCenterR - pyramidCenterG,
                                                                  pyramidScaleG - pyramidScaleR))
                for item in centerOp:
                    RG.append(item)
                centerOp = (self.__CenterSurroundOperatorAbsolute(pyramidCenterB - pyramidCenterY,
                                                                  pyramidScaleY - pyramidScaleB))
                for item in centerOp:
                    BY.append(item)

        return RG, BY

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Orientation
    # feature maps, O(c; s; ), encode, as a group, local orientation
    # contrast between the center and surround scales:
    # O(c; s; )=jO(c; )
    def __BuildOrientationFeatureMaps(self) -> list:
        orientationFeatureMaps = []

        for delta in self.__deltas:
            for center in self.__centers:
                for angle in self.__angles:
                    # They may have to be built from Intensity type image tho
                    pyramidCenter = self.__BuildOrientationPyramid(
                        center, angle)
                    pyramidScale = self.__BuildOrientationPyramid(
                        center + delta, angle)
                    centerOp = (
                        self.__CenterSurroundOperatorAbsolute(pyramidCenter, pyramidScale))
                    for item in centerOp:
                        orientationFeatureMaps.append(item)

        return orientationFeatureMaps

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Center-surround is implemented in the model as the dierence between ne and
    # coarse scales: The center is a pixel at scale c {2; 3; 4} ,and
    # the surround is the corresponding pixel at scale s = c + delta, with
    # delta {3; 4}
    def __CenterSurroundOperatorAbsolute(self, a: GaussianPyramid, b: GaussianPyramid) -> list:
        higher_pyramid, lower_pyramid = (
            a, b) if a.height > b.height else (b, a)
        higher_pyramid_layers = higher_pyramid.layers[higher_pyramid.height -
                                                      lower_pyramid.height:]
        input_size = lower_pyramid.original.shape
        higher_pyramid_layers = self.__upscale(
            higher_pyramid_layers, input_size)
        lower_pyramid_layers = self.__upscale(lower_pyramid.layers, input_size)

        return [abs(ll - hl) for ll, hl in zip(lower_pyramid_layers, higher_pyramid_layers)]

    def __upscale(self, layers: list, dsize=()) -> list:
        return [cv2.resize(layer, (dsize[1], dsize[0])) for layer in layers]

    def __NormalizeOperator(self, image: np.ndarray) -> np.ndarray:
        max = image[0][0]
        for i in range(0, len(image)):
            for j in range(0, len(image[i])):
                if image[i][j] > max:
                    max = image[i][j]

        if max == 0:
            max = 1

        for i in range(0, len(image)):
            for j in range(0, len(image[i])):
                image[i][j] /= max

        return image

    # def __AcrossScaleAdditionOperatorIntensity(self, featureMaps: list) -> np.ndarray:
    #     output = np.zeros(np.array(featureMaps[0]).shape)
    #     outputFlat = np.zeros(np.array(featureMaps[0]).shape[:2])
    #
    #     # finding local maxes in each feature map
    #     localMaxes = []
    #     globalMax = map[0][0][0]
    #     for map in featureMaps:
    #         localMax = map[0][0][0]
    #         for i in range(0, len(map)):
    #             for j in range(0, len(map)):
    #                 if map[i][j][0] > localMax:
    #                     localMax = map[i][j][0]
    #
    #         # adding local maxes to the list
    #         localMaxes.append(localMax)
    #         if localMax > globalMax:
    #             globalMax = localMax
    #
    #     # mean of local maxes and (M-m)^2
    #     meanlocalMax = sum(localMaxes) / len(localMaxes)
    #     factor = pow((globalMax - meanlocalMax), 2)
    #
    #     # normalization to global max M
    #     for map in featureMaps:
    #         for i in range(0, len(map)):
    #             for j in range(0, len(map)):
    #                 map[i][j][0] /= globalMax
    #
    #     # global multiplication of the map
    #     for map in featureMaps:
    #         for i in range(0, len(map)):
    #             for j in range(0, len(map)):
    #                 map[i][j][0] *= factor
    #
    #     # across-scale point by point addition
    #     for i in range(0, len(output)):
    #         for j in range(0, len(output[i])):
    #             for map in featureMaps:
    #                 output[i][j][0] += map[i][j][0]
    #
    #     # to flat
    #     for i in range(0, len(output)):
    #         for j in range(0, len(output)):
    #             outputFlat[i][j] = output[i][j][0]
    #
    #     return outputFlat

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (5)
    def __BuildConspicuityInensityMap(self) -> np.ndarray:
        featureMaps = self.__intensityFeatureMaps
        output = np.zeros(np.array(featureMaps[0]).shape)
        outputFlat = np.zeros(np.array(featureMaps[0]).shape[:2])

        # finding local maxes in each feature map
        localMaxes = []
        globalMax = featureMaps[0][0][0]
        for map in featureMaps:
            localMax = map[0][0]
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    if map[i][j] > localMax:
                        localMax = map[i][j]

            # adding local maxes to the list
            localMaxes.append(localMax)
            if localMax > globalMax:
                globalMax = localMax

        # mean of local maxes and (M-m)^2
        meanlocalMax = sum(localMaxes) / len(localMaxes)
        factor = pow((globalMax - meanlocalMax), 2)

        # normalization to global max M
        for map in featureMaps:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] /= globalMax

        # global multiplication of the map
        for map in featureMaps:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] *= factor

        # across-scale point by point addition
        for i in range(0, len(output)):
            for j in range(0, len(output[i])):
                for map in featureMaps:
                    output[i][j] += map[i][j]

        # to flat
        for i in range(0, len(output)):
            for j in range(0, len(output[i])):
                outputFlat[i][j] = output[i][j]

        return outputFlat

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (6)
    def __BuildConspicuityColorMap(self) -> np.ndarray:
        # for RG
        featureMapsRG = self.__colorFeatureMaps[0]
        output = np.zeros(np.array(featureMapsRG[0]).shape)
        outputFlat = np.zeros(np.array(featureMapsRG[0]).shape[:2])

        # finding local maxes in each feature map
        localMaxes = []
        globalMax = featureMapsRG[0][0][0]
        for map in featureMapsRG:
            localMax = map[0][0]
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    if map[i][j] > localMax:
                        localMax = map[i][j]

            # adding local maxes to the list
            localMaxes.append(localMax)
            if localMax > globalMax:
                globalMax = localMax

        # mean of local maxes and (M-m)^2
        meanlocalMax = sum(localMaxes) / len(localMaxes)
        factor = pow((globalMax - meanlocalMax), 2)

        # normalization to global max M
        for map in featureMapsRG:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] /= globalMax

        # global multiplication of the map
        for map in featureMapsRG:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] *= factor

        ###############################################################################################################
        # for BY
        featureMapsBY = self.__colorFeatureMaps[1]
        output = np.zeros(np.array(featureMapsBY[0]).shape)
        outputFlat = np.zeros(np.array(featureMapsBY[0]).shape[:2])

        # finding local maxes in each feature map
        localMaxes = []
        globalMax = featureMapsBY[0][0][0]
        for map in featureMapsBY:
            localMax = map[0][0]
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    if map[i][j] > localMax:
                        localMax = map[i][j]

            # adding local maxes to the list
            localMaxes.append(localMax)
            if localMax > globalMax:
                globalMax = localMax

        # mean of local maxes and (M-m)^2
        meanlocalMax = sum(localMaxes) / len(localMaxes)
        factor = pow((globalMax - meanlocalMax), 2)

        # normalization to global max M
        for map in featureMapsBY:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] /= globalMax

        # global multiplication of the map
        for map in featureMapsBY:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] *= factor

        # across-scale point by point addition
        sumMap = [a + b for a, b in zip(featureMapsRG, featureMapsBY)]
        for i in range(0, len(output)):
            for j in range(0, len(output[i])):
                for map in sumMap:
                    output[i][j] += map[i][j]

        # to flat
        for i in range(0, len(output)):
            for j in range(0, len(output[i])):
                outputFlat[i][j] = output[i][j]

        return outputFlat

    # A Model of Saliency-based Visual Attention for Rapid Scene Analysis:
    # Page 2, equation (7)

    def __BuildConspicuityOrientationMap(self) -> np.ndarray:
        featureMaps = self.__orientationFeatureMaps
        output = np.zeros(np.array(featureMaps[0]).shape)
        outputFlat = np.zeros(np.array(featureMaps[0]).shape[:2])

        # finding local maxes in each feature map
        localMaxes = []
        globalMax = featureMaps[0][0][0]
        for map in featureMaps:
            localMax = map[0][0]
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    if map[i][j] > localMax:
                        localMax = map[i][j]

            # adding local maxes to the list
            localMaxes.append(localMax)
            if localMax > globalMax:
                globalMax = localMax

        # mean of local maxes and (M-m)^2
        meanlocalMax = sum(localMaxes) / len(localMaxes)
        factor = pow((globalMax - meanlocalMax), 2)

        # normalization to global max M
        for map in featureMaps:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] /= globalMax

        # global multiplication of the map
        for map in featureMaps:
            for i in range(0, len(map)):
                for j in range(0, len(map[i])):
                    map[i][j] *= factor

        # across-scale point by point addition
        for i in range(0, len(output)):
            for j in range(0, len(output[i])):
                for map in featureMaps:
                    output[i][j] += map[i][j]

        # to flat
        for i in range(0, len(output)):
            for j in range(0, len(output[i])):
                outputFlat[i][j] = output[i][j]

        return outputFlat

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
                s[i][j] = (nIDash[i][j] + nCDash[i][j] + nODash[i][j]) / 3
        return s
