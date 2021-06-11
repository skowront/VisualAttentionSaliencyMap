import numpy as np
import cv2
from numpy.core.fromnumeric import resize


class GaussianPyramid:
    def __init__(self, original: np.ndarray, pyramidHeight: int = 1) -> None:
        self.__layers: list[np.ndarray] = list()
        self.__height: int = pyramidHeight
        self.__original = original
        self.__BuildPyramids()

    @property
    def original(self) -> list[np.ndarray]:
        return self.__original

    @original.setter
    def original(self, value):
        self.__original = value
        self.__BuildPyramids()
        return

    @property
    def layers(self) -> list[np.ndarray]:
        return self.__layers

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, value):
        self.__height = value
        self.__BuildPyramids()
        return

    def __BuildPyramids(self):
        self.__layers = list()
        resized = self.__original
        for i in range(0, self.__height):
            self.__layers.append(resized)
            newDim = (int(resized.shape[1]/2), int(resized.shape[0]/2))
            resized = cv2.resize(src=resized, dsize=newDim,
                                 interpolation=cv2.INTER_CUBIC)

    def ShowPyramid(self):
        layers = self.layers
        sumWidth = 0
        sumHeight = self.layers[0].shape[0]
        for layer in layers:
            sumWidth = sumWidth + layer.shape[1]
        if len(layer.shape) <= 2:
            out = np.zeros((sumHeight, sumWidth))
        else:
            out = np.zeros((sumHeight, sumWidth, len(layers[0][0][0])))
        offset = 0
        for layer in layers:
            for i in range(0, len(layer)):
                for j in range(0, len(layer[i])):
                    pixel = layer[i][j]
                    if len(layer.shape) <= 2:
                        out[i][offset+j] = float(pixel)
                    else:
                        if len(pixel) > 0:
                            out[i][offset+j][0] = float(pixel[0])
                        if len(pixel) > 1:
                            out[i][offset+j][1] = float(pixel[1])
                        if len(pixel) > 2:
                            out[i][offset+j][2] = float(pixel[2])
            offset = offset + layer.shape[1]
        cv2.imshow('Gaussian Pyramid', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
