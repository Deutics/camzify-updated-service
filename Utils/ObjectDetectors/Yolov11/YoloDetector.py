"""Standalone YOLO Detector (No Django dependencies)"""

from ultralytics import YOLO
from Utils.logger import get_logger
import os
import torch

logger = get_logger(__name__)


def get_indexes(model_classes, expected_classes):
    """Find the indexes of expected_classes from model_classes"""
    if isinstance(model_classes, dict):
        model_classes = list(model_classes.values())

    indexes = []
    for label in expected_classes:
        if label in model_classes:
            indexes.append(model_classes.index(label))

    return indexes


class YoloDetector:
    def __init__(self, model_name="yolo11n.pt", model_path=None, conf_threshold=0.45,
                 iou_thresh=0.7, gpu_inference=False, expected_objs=None):
        """
        Initialize YOLO Detector
        
        Args:
            model_name: Name of the YOLO model file
            model_path: Full path to model (if None, uses model_name from current dir or downloads)
            conf_threshold: Confidence threshold for detections
            iou_thresh: IOU threshold for NMS
            gpu_inference: Use GPU if available
            expected_objs: List of class names to detect (None = detect all)
        """
        
        if model_path:
            self._weights_file_path = model_path
        else:
            # Check if model exists in current directory
            if os.path.exists(model_name):
                self._weights_file_path = model_name
            else:
                # Let ultralytics download it automatically
                self._weights_file_path = model_name
        self._model = YOLO(self._weights_file_path)
        
        self._conf_thresh = conf_threshold
        self._iou_thresh = iou_thresh
        self._device = 0 if gpu_inference and torch.cuda.is_available() else "cpu"
        logger.info(f"Loading weights from: {self._weights_file_path} and device: {self._device}")
        
        self._names = self._model.names
        self._expected_objs = get_indexes(self._names, expected_classes=expected_objs) \
            if isinstance(expected_objs, list) else None

    def process_frame(self, frame):
        """
        Process frame and return detections
        
        Returns:
            List of dicts with 'label', 'bbox', 'confidence'
        """
        results = self._model.predict(source=frame, conf=self._conf_thresh, classes=self._expected_objs,
                                      device=self._device, verbose=False, iou=self._iou_thresh)
        detections = results[0].boxes.data

        detected_objects = []

        for detection in detections:
            detection = detection.cpu()
            bbox = detection[:4].tolist()
            confidence = float(detection[4])
            label = self._names[int(detection[5])]
            detected_objects.append({
                'label': label,
                'bbox': bbox,
                'confidence': confidence
            })

        return detected_objects

    def process_image_for_tracker(self, frame):
        """
        Process frame and return raw boxes object for tracker
        
        Returns:
            Ultralytics boxes object
        """
        results = self._model.predict(source=frame, conf=self._conf_thresh, classes=self._expected_objs,
                                      device=self._device, verbose=False, iou=self._iou_thresh)
        return results[0].boxes

    @property
    def names(self):
        return self._names
