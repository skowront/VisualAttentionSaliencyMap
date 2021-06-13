class Neuron:
    def __init__(self, startValue: float, endValue: float, regenerationTime) -> None:
        self.__startValue = startValue
        self.__endValue = endValue
        self.__regenerationTime = regenerationTime
        self.__currentValue = self.__startValue
        pass

    @property
    def CurrentValue(self):
        return self.__currentValue

    # Sets the neuron's value to startValue
    # If you want to take into account some distance factor, please use factor argument so the neuron's substracted value will be scaled appropriately.
    def Zero(self, factor: float = 1.0):
        self.__currentValue = self.__currentValue - \
            (factor*abs(self.__currentValue-self.__startValue))

    # Regenerates the value of the pixel. That is not a limes function, but a simple portion addition.
    def Regenerate(self):
        self.__currentValue = self.__currentValue + \
            (self.__endValue/self.__regenerationTime)
        if self.__currentValue > self.__endValue:
            self.__currentValue = self.__endValue

    # Sets the neuron's value to endValue
    def Fill(self):
        self.__currentValue = self.__endValue
