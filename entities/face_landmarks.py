import numpy as np

from entities import Point


class FaceLandmarks:

    def __init__(
            self,
            right_eye: Point,
            left_eye: Point,
            nose: Point,
            right_mouth_corner: Point,
            left_mouth_corner: Point,
    ):
        self.right_eye = right_eye
        self.left_eye = left_eye
        self.nose = nose
        self.right_mouth_corner = right_mouth_corner
        self.left_mouth_corner = left_mouth_corner


    @classmethod
    def from_list(cls, array: np.ndarray):
        return FaceLandmarks(
            right_eye=Point(x=array[0], y=array[1]),
            left_eye=Point(x=array[2], y=array[3]),
            nose=Point(x=array[4], y=array[5]),
            right_mouth_corner=Point(x=array[6], y=array[7]),
            left_mouth_corner=Point(x=array[8], y=array[9])
        )