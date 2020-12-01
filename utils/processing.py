from typing import List, Tuple

import numpy as np
import cv2

from entities import BoundingBox, FaceLandmarks


def blur(
        image: np.ndarray,
        bboxes: List[BoundingBox],
        ksize: Tuple[int, int] = (30, 30)
) -> np.ndarray:

    image_copy = np.copy(image)
    for bbox in bboxes:
        crop = image_copy[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
        crop = cv2.blur(crop, ksize, cv2.BORDER_TRANSPARENT)
        image_copy[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] = crop

    return image_copy


def pixelate(
        image: np.ndarray,
        bboxes: List[BoundingBox],
        size: Tuple[int, int] = (16, 16)
) -> np.ndarray:

    image_copy = np.copy(image)
    for bbox in bboxes:
        crop = image_copy[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
        height, width = crop.shape[:2]
        crop_small = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
        crop_pix = cv2.resize(crop_small, (width, height), interpolation=cv2.INTER_NEAREST)
        image_copy[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] = crop_pix

    return image_copy


def hide_eyes(
        image: np.ndarray,
        landmarks: List[FaceLandmarks]
) -> np.ndarray:

    image_copy = np.copy(image)
    for landm in landmarks:
        xmin, ymin = landm.left_eye.to_tuple()
        xmax, ymax = landm.right_eye.to_tuple()
        cv2.rectangle(image_copy, (xmin - 25, ymin - 20), (xmax + 25, ymax + 20), (0, 0, 0), thickness=cv2.FILLED)

    return image_copy
