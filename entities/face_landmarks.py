from entities import Point


class FaceLandmarks:

    def __init__(
            self,
            left_eye: Point,
            right_eye: Point,
            nose: Point,
            left_mouth_corner: Point,
            right_mouth_corner: Point
    ):
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose = nose
        self.left_mouth_corner = left_mouth_corner
        self.right_mouth_corner = right_mouth_corner

    @classmethod
    def from_list(cls, array: int):
        return FaceLandmarks(
            left_eye=Point(x=array[0], y=array[1]),
            right_eye=Point(x=array[2], y=array[3]),
            nose=Point(x=array[4], y=array[5]),
            left_mouth_corner=Point(x=array[6], y=array[7]),
            right_mouth_corner=Point(x=array[8], y=array[9])
        )