class BoundingBox:

    def __init__(
            self,
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @classmethod
    def from_list(cls, array):
        return BoundingBox(
            xmin=array[0],
            ymin=array[1],
            xmax=array[2],
            ymax=array[3]
        )