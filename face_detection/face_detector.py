import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn import Module
from typing import List, Optional

from face_detection.models.retinaface import RetinaFace
from face_detection.data import cfg_re50
from face_detection.layers.functions.prior_box import PriorBox
from face_detection.utils.box_utils import decode, decode_landm
from face_detection.utils.nms.py_cpu_nms import py_cpu_nms
from face_detection.detection_config import *
from helpers.detection import load_model
from helpers.ml import DeviceType
from shapes import BoundingBox


class FaceDetector:

    def __init__(self, model: Module, device: torch.device):
        self.device = device
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> List[BoundingBox]:
        image = np.float32(image)
        image_height, image_width, _ = image.shape
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)
        scale = scale.to(self.device)

        loc, conf, _ = self.model(image)  # forward pass

        priorbox = PriorBox(cfg_re50, image_size=(image_height, image_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > CONFIDENCE)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:TOP_K]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, NMS_THRESHOLD)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:KEEP_TOP_K, :]

        det_bboxes = []
        for det in dets:
            det_bboxes.append(BoundingBox(det[0], det[1], det[2], det[3]))

        return det_bboxes

    @classmethod
    def from_path(cls, path: str, device_type: Optional[DeviceType] = None):
        net = RetinaFace(cfg=cfg_re50, phase='test')
        if device_type is None:
            device_type = DeviceType.cuda if torch.cuda.is_available() else DeviceType.cpu
        device = device_type.get()

        load_to_cpu = False
        if device_type == DeviceType.cpu: load_to_cpu = True

        model = load_model(
            model=net, pretrained_path=path, load_to_cpu=load_to_cpu
        )
        model.eval()
        cudnn.benchmark = True

        return FaceDetector(model=model, device=device)

