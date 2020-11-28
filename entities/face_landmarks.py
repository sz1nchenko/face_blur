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