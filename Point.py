class Point:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.__x = x
        self.__y = y

    @property
    def X(self):
        return self.__x

    @property
    def Y(self):
        return self.__y

    def Print(self):
        print("Point x="+str(self.__x)+" y="+str(self.__y))

    def __str__(self) -> str:
        return "Point x = "+str(self.__x)+" y ="+str(self.__y)
