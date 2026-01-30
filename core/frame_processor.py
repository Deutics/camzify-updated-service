"""Core Frame Processor - Single detection, per-stream tracking"""

import cv2
import numpy as np
from typing import Dict, List
from Utils.ObjectDetectors.Yolov11.YoloDetector import YoloDetector
from Utils.Trackers.Sort.sort.sort import Sort
from Utils.logger import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Central processor - ONE model for detection, SEPARATE tracking per stream"""
    
    def __init__(self, model_path: str = "yolo11n.pt", conf_thresh: float = 0.45):
        """
        Initialize single detector for ALL streams, separate trackers per stream
        
        Args:
            model_path: Path to YOLO model
            conf_thresh: Detection confidence threshold
        """
        logger.info("Initializing FrameProcessor with single model for all streams")
        
        # ONE YOLO Detector for all streams (shared)
        # NOTE: Must detect ALL classes - filtering happens at feature level
        self.yolo_detector = YoloDetector(
            model_name="yolo11n.pt",
            model_path=model_path,
            gpu_inference=True,
            conf_threshold=conf_thresh,
            expected_objs=None  # Detect all classes, filter per-stream
        )
        
        # SEPARATE SORT trackers per stream (lightweight, no models)
        self.stream_trackers: Dict[str, Sort] = {}
        
        # Tracker configuration
        self.tracker_max_age = 1
        self.tracker_min_hits = 3
        self.tracker_iou_threshold = 0.5
        
        logger.info("FrameProcessor initialized - single YOLO model, per-stream SORT trackers")
    
    def _get_tracker(self, stream_id: str) -> Sort:
        """Get or create SORT tracker for specific stream"""
        if stream_id not in self.stream_trackers:
            self.stream_trackers[stream_id] = Sort(
                max_age=self.tracker_max_age,
                min_hits=self.tracker_min_hits,
                iou_threshold=self.tracker_iou_threshold
            )
            logger.info(f"Created SORT tracker for stream: {stream_id}")
        return self.stream_trackers[stream_id]
    
    def process_frame(self, frame: np.ndarray, stream_id: str, features: List = None):
        """
        Process frame: detect once (shared), track per-stream, features classify
        
        Args:
            frame: Input frame
            stream_id: Stream identifier
            features: List of feature instances for this stream
            
        Returns:
            annotated_frame: Frame with visualizations
        """
        annotated_frame = frame.copy()
        
        # 1. SINGLE DETECTION (shared YOLO for all streams - detects ALL classes)
        detections_boxes = self.yolo_detector.process_image_for_tracker(frame)
        
        # 2. PER-STREAM TRACKING (separate SORT tracker per stream)
        tracker = self._get_tracker(stream_id)
        
        # Convert detections to format SORT expects
        if detections_boxes.data is not None and len(detections_boxes.data) > 0:
            dets = detections_boxes.data.cpu().numpy()
        else:
            dets = np.empty((0, 6))
        
        # Update tracker for this stream
        tracked_detections = tracker.update(dets)
        
        # Get tracks with position history (for line crossing)
        tracks = tracker.get_tracks()
        
        # Convert tracked detections to dict format for visualization
        tracked_objects = []
        for det in tracked_detections:
            tracked_objects.append({
                "bbox": det[:4].tolist(),
                "tracker_id": int(det[4]),
                "label": self.yolo_detector.names[int(det[5])] if len(det) > 5 else "unknown",
                "confidence": det[4] if len(det) > 4 else 0.0
            })
        
        # 3. FEATURES CLASSIFY (with tracked_objects for class info)
        if features:
            for feature in features:
                try:
                    # Pass tracked_objects for simple class lookup
                    feature.process(tracks, annotated_frame, tracked_objects)
                except Exception as e:
                    logger.error(f"Feature {type(feature).__name__} error: {e}", exc_info=True)
        
        # 4. Draw basic visualization (bbox + track IDs)
        annotated_frame = self._draw_tracked_objects(annotated_frame, tracked_objects)
        
        return annotated_frame
    
    def _draw_tracked_objects(self, frame: np.ndarray, tracked_objects: List):
        """Draw simple bounding boxes for tracked objects"""
        for obj in tracked_objects:
            bbox = obj["bbox"]
            tracker_id = int(obj["tracker_id"])
            label = obj.get("label", "unknown")
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            text = f"ID:{tracker_id} {label}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        return frame
