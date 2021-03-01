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
        size: Tuple[int, int] = (10, 10)
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
        right_eye = landm.right_eye
        left_eye = landm.left_eye
        dist = right_eye.distance(left_eye)
        x_pad = int(dist * 0.4)
        y_pad = int(dist * 0.3)
        p1 = [right_eye.x - x_pad, right_eye.y - y_pad]
        p2 = [right_eye.x - x_pad, right_eye.y + y_pad]
        p3 = [left_eye.x + x_pad, left_eye.y - y_pad]
        p4 = [left_eye.x + x_pad, left_eye.y + y_pad]

        cnt = np.array([p1, p2, p3, p4])
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image_copy, [box], 0, (0, 0, 0), -1)

    return image_copy


