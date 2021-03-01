import os
import cv2
from argparse import ArgumentParser

from face_detection import FaceDetector
from utils.effects import blur, pixelate, hide_eyes


IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']
VIDEO_FORMATS = ['.mpeg', '.mp4', '.avi', '.mkv', '.mov']

def process_image(
    detector: FaceDetector,
    filepath: str,
    mode: str = 'blur',
):
    image = cv2.imread(filepath)
    bboxes, landmarks = detector.predict(image)

    if mode == 'blur':
        image = blur(image, bboxes)
    elif mode == 'pixel':
        image = pixelate(image, bboxes)
    else:
        image = hide_eyes(image, landmarks)

    os.makedirs('./output', exist_ok=True)
    output_path = os.path.join('./output', os.path.split(args.file)[1])
    cv2.imwrite(output_path, image)


def process_video(
    detector: FaceDetector,
    filepath: str,
    mode: str = 'blur',
):
    cap = cv2.VideoCapture(filepath)
    _, frame = cap.read()
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('./output/video.avi', fourcc, 20, (width, height), True)

    while cap.isOpened():
        _, frame = cap.read()
        bboxes, landmarks = detector.predict(frame)

        if mode == 'blur':
            frame = blur(frame, bboxes)
        elif mode == 'pixel':
            frame = pixelate(frame, bboxes)
        else:
            frame = hide_eyes(frame, landmarks)

        writer.write(frame)

    cap.release()
    writer.release()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-m', '--mode', choices=['blur', 'pixel', 'eyes'], default='blur', required=True)
    parser.add_argument('-w', '--weights', default='./weights/Resnet50_Final.pth')

    args = parser.parse_args()

    detector = FaceDetector.from_path(args.weights)

    ext = os.path.splitext(args.file)[1]
    if ext in IMAGE_FORMATS:
        process_image(detector=detector, filepath=args.file, mode=args.mode)
    elif ext in VIDEO_FORMATS:
        process_video(detector=detector, filepath=args.file, mode=args.mode)
    else:
        raise Exception(f'Unknown file format: {ext}')

