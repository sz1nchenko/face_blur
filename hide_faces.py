import os
import cv2
from argparse import ArgumentParser

from face_detection import FaceDetector
from utils.drawing import draw_bbox, draw_landmarks
from utils.processing import blur, pixelate, hide_eyes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    print('Reading image ...')
    image = cv2.imread(args.file)

    print('Loading detector')
    detector = FaceDetector.from_path('./weights/detection_model.pth')
    print('Predicting ...')
    bboxes, landmarks = detector.predict(image)
    image = hide_eyes(image, landmarks)
    # for bbox, landm in zip(bboxes, landmarks):
    #     image = draw_bbox(image, bbox)
    #     image = draw_landmarks(image, landm)

    os.makedirs('./output', exist_ok=True)
    output_path = os.path.join('./output', os.path.split(args.file)[1])
    cv2.imwrite(output_path, image)