"""Object Tracker combining YOLO detection with SORT tracking"""

from .sort.sort import *
from .sort.utils import *
from ...ObjectDetectors.Yolov11.YoloDetector import YoloDetector


class ObjectTracker:
    def __init__(self,
                 detector_model="yolo11n.pt",
                 model_path=None,
                 obj_size=None,
                 conf_thresh=0.6,
                 iou_thresh=0.6,
                 gpu_inference=False,
                 expected_objs=None,
                 tracker_max_age=1,
                 tracker_min_hits=3,
                 tracker_iou_threshold=0.5):
        """
        Initialize Object Tracker
        
        Args:
            detector_model: YOLO model name
            model_path: Full path to model (optional, if None uses detector_model)
            obj_size: Dict with Width, Height, Precision for size filtering
            conf_thresh: Confidence threshold for detection
            iou_thresh: IOU threshold for detection NMS
            gpu_inference: Use GPU if available
            expected_objs: List of class names to track
            tracker_max_age: Max frames to keep track without detection
            tracker_min_hits: Min detections before track is confirmed
            tracker_iou_threshold: IOU threshold for track matching
        """

        self.yolo_detector = YoloDetector(
            model_name=detector_model,
            model_path=model_path,
            gpu_inference=gpu_inference,
            conf_threshold=conf_thresh,
            expected_objs=expected_objs,
            iou_thresh=iou_thresh
        )

        self._tracker = Sort(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou_threshold
        )
        
        self._object_size_thresholds = self.calculate_size_bounds(obj_size)

    def process_frame(self, frame):
        """
        Process frame through detection and tracking
        
        Returns:
            List of tracked objects (empty list if no detections)
        """
        predictions_from_detector = self.yolo_detector.process_image_for_tracker(frame).data
        tracked_objects = []

        if len(predictions_from_detector):
            updated_detections = self.transform_detections(predictions_from_detector)

            if updated_detections:
                tracker_detections = self._tracker.update(np.array(updated_detections))

                for detection in tracker_detections:
                    temp = {
                        "bbox": detection[:4].tolist(),
                        "label": self.model_classes()[int(detection[5])],
                        "tracker_id": detection[4]
                    }
                    tracked_objects.append(temp)

        return tracked_objects

    def transform_detections(self, detections):
        """Filter detections by size if configured"""
        detections = detections.cpu().tolist()
        transformed_detections = []

        for detection in detections:
            if self._is_not_outlier_detection(detection[:4]):
                transformed_detections.append(detection)

        return transformed_detections

    def _is_not_outlier_detection(self, detection):
        """Check if detection size is within configured bounds"""
        box = xyxy2xywh(detection)

        if self._object_size_thresholds is not None:
            # Check if the box dimensions are within the valid range
            return (self._object_size_thresholds["min_width"] <= box[2] <= self._object_size_thresholds["max_width"]) \
                and (self._object_size_thresholds["min_height"] <= box[3] <= self._object_size_thresholds["max_height"])
        return True

    @staticmethod
    def calculate_size_bounds(object_size):
        """Calculate min/max size thresholds from config"""
        object_size_thresholds = None

        if object_size is not None:
            precision_factor = int(object_size["Precision"]) * 0.01

            max_width = int(object_size["Width"]) + (int(object_size["Width"]) * precision_factor)
            max_height = int(object_size["Height"]) + (int(object_size["Height"]) * precision_factor)

            min_width = int(object_size["Width"]) - (int(object_size["Width"]) * precision_factor)
            min_height = int(object_size["Height"]) - (int(object_size["Height"]) * precision_factor)

            object_size_thresholds = {
                "max_width": max_width,
                "max_height": max_height,
                "min_width": min_width,
                "min_height": min_height
            }

        return object_size_thresholds

    @property
    def tracker(self):
        """Get the underlying SORT tracker (to access tracks)"""
        return self._tracker

    def model_classes(self):
        """Get YOLO model class names"""
        return self.yolo_detector.names
