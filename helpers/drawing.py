import numpy as np
import cv2

from entities import BoundingBox, FaceLandmarks


def draw_bbox(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    image_copy = np.copy(image)
    cv2.rectangle(image_copy, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (124, 252, 0), 2)

    return image_copy


def draw_landmarks(image: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
    image_copy = np.copy(image)
    cv2.circle(image_copy, landmarks.left_eye.to_tuple(), 1, (255, 140, 0), 4)
    cv2.circle(image_copy, landmarks.right_eye.to_tuple(), 1, (255, 140, 0), 4)
    cv2.circle(image_copy, landmarks.nose.to_tuple(), 1, (255, 140, 0), 4)
    cv2.circle(image_copy, landmarks.left_mouth_corner.to_tuple(), 1, (255, 140, 0), 4)
    cv2.circle(image_copy, landmarks.right_mouth_corner.to_tuple(), 1, (255, 140, 0), 4)

    return image_copy
