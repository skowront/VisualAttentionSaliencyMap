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

    def Zero(self):
        self.__currentValue = self.__startValue

    def Regenerate(self):
        self.__currentValue = self.__currentValue + \
            (self.__endValue/self.__regenerationTime)
        if self.__currentValue > self.__endValue:
            self.__currentValue = self.__endValue

    def Fill(self):
        self.__currentValue = self.__endValue
