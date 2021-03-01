from typing import Tuple

import numpy as np


class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def distance(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)