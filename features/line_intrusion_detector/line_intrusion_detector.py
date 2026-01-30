"""
Line Intrusion Detector - Simple class filtering

Pattern matches pose-recognition: internal notification handling, direct class check
"""

import cv2
import asyncio
from datetime import datetime
from typing import List, Tuple, Optional
from Utils.Line.Line import Line
from notification.local_notification_handler import Notification
from Utils.logger import get_logger

logger = get_logger(__name__)


class LineIntrusionDetector:
    """Line intrusion classifier - checks if tracked objects cross line"""
    
    # Class-level shared async notification handler (initialized once)
    _shared_async_handler = None
    _async_handler_initialized = False
    
    def __init__(
        self,
        line_coords: List[Tuple[int, int]],
        instance_id: str,
        stream_id: str,
        direction_to_use: int = 0,
        bounding_box_start: Optional[Tuple[int, int]] = None,
        bounding_box_end: Optional[Tuple[int, int]] = None,
        precision_factor: int = 0,
        time_bound_start: str = "00:00:00",
        time_bound_end: str = "23:59:59",
        alert_classes: Optional[List[str]] = None,
        use_async_notifications: bool = False,
    ):
        """
        Args:
            line_coords: [(x1, y1), (x2, y2)]
            instance_id: Unique instance ID
            stream_id: Stream ID
            direction_to_use: 0=any, 1=left, 2=right
            bounding_box_start: (x, y) for reference object size
            bounding_box_end: (x, y) for reference object size
            precision_factor: Percentage tolerance for object size (0-100)
            time_bound_start: "HH:MM:SS"
            time_bound_end: "HH:MM:SS"
            alert_classes: List of object classes to detect (e.g., ["person", "car"])
            use_async_notifications: Use async handler (True) or sync local (False)
        """
        self._instance_id = instance_id
        self._stream_id = stream_id
        self._line = Line(start_position=line_coords[0], end_position=line_coords[1])
        
        # Direction check: 0=any, 1=left, 2=right
        self._direction_to_use = direction_to_use
        
        # Time bounds
        self._time_bound_start = time_bound_start
        self._time_bound_end = time_bound_end
        
        # Object size thresholds
        self._size_thresholds = self._calculate_size_thresholds(
            bounding_box_start, bounding_box_end, precision_factor
        )
        
        # Alert classes - converted to set for O(1) lookup
        self._alert_classes = set(alert_classes) if alert_classes else None
        
        # Notification mode
        self._use_async = use_async_notifications
        
        # Initialize shared async handler if needed (lazy initialization)
        if self._use_async and not LineIntrusionDetector._async_handler_initialized:
            self._initialize_async_handler()
        
        logger.info(f"[LineIntrusion:{stream_id}:{instance_id}] Initialized")
        logger.info(f"  Direction: {direction_to_use} (0=any, 1=left, 2=right)")
        logger.info(f"  Time: {time_bound_start} - {time_bound_end}")
        logger.info(f"  Size filter: {self._size_thresholds is not None}")
        logger.info(f"  Alert classes: {alert_classes}")
        logger.info(f"  Notification mode: {'async' if use_async_notifications else 'sync'}")

    @classmethod
    def _initialize_async_handler(cls):
        """Initialize shared async notification handler once for all instances"""
        if not cls._async_handler_initialized:
            try:
                from notification.async_notification_handler import AsyncNotificationHandler
                cls._shared_async_handler = AsyncNotificationHandler(
                    output_dir="output/vaf_alerts",
                    queue_size=100,
                    workers=2
                )
                # Start the handler
                asyncio.create_task(cls._shared_async_handler.start())
                cls._async_handler_initialized = True
                logger.info("Shared AsyncNotificationHandler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize async handler: {e}")
                cls._shared_async_handler = None
                cls._async_handler_initialized = False

    def _calculate_size_thresholds(self, bbox_start, bbox_end, precision_factor):
        """Calculate min/max size thresholds from bounding box and precision factor"""
        if not bbox_start or not bbox_end or precision_factor == 0:
            return None
        
        ref_width = abs(bbox_end[0] - bbox_start[0])
        ref_height = abs(bbox_end[1] - bbox_start[1])
        
        if ref_width == 0 or ref_height == 0:
            return None
        
        precision = precision_factor * 0.01
        
        min_width = ref_width - (ref_width * precision)
        max_width = ref_width + (ref_width * precision)
        min_height = ref_height - (ref_height * precision)
        max_height = ref_height + (ref_height * precision)
        
        return {
            "min_width": min_width,
            "max_width": max_width,
            "min_height": min_height,
            "max_height": max_height
        }

    def _is_within_time_bounds(self) -> bool:
        """Check if current time is within configured bounds"""
        try:
            current_time = datetime.now().time()
            start_time = datetime.strptime(self._time_bound_start, "%H:%M:%S").time()
            end_time = datetime.strptime(self._time_bound_end, "%H:%M:%S").time()
            
            return start_time <= current_time <= end_time
        except Exception as e:
            logger.error(f"Time bound check failed: {e}")
            return True

    def _is_valid_object_size(self, bbox: List[float]) -> bool:
        """Check if detected object size is within configured thresholds"""
        if self._size_thresholds is None:
            return True
        
        obj_width = bbox[2] - bbox[0]
        obj_height = bbox[3] - bbox[1]
        
        return (
            self._size_thresholds["min_width"] <= obj_width <= self._size_thresholds["max_width"] and
            self._size_thresholds["min_height"] <= obj_height <= self._size_thresholds["max_height"]
        )

    def _matches_direction(self, angle: float) -> bool:
        """Check if crossing direction matches configuration"""
        if self._direction_to_use == 0:
            return True
        
        is_left = angle > 0
        
        if self._direction_to_use == 1:
            return is_left
        else:
            return not is_left

    def process(self, tracks: List, frame, tracked_objects: List[dict]):
        """
        Check for line intrusions with all filters applied
        
        Args:
            tracks: List of KalmanBoxTracker objects from SORT (for position history)
            frame: Frame for visualization
            tracked_objects: List of dicts with 'label', 'bbox', 'tracker_id' from frame_processor
        """
        # Draw line
        cv2.line(frame, self._line.start_position, self._line.end_position, (0, 0, 255), 2)
        
        # Check time bounds
        if not self._is_within_time_bounds():
            return
        
        # Create lookup map: tracker_id -> tracked_object (for class info)
        track_map = {obj["tracker_id"]: obj for obj in tracked_objects}
        
        # Check each track
        for track in tracks:
            track_id = track.track_id + 1
            
            # Get corresponding tracked object for class info
            if track_id not in track_map:
                continue
            
            tracked_obj = track_map[track_id]
            class_label = tracked_obj["label"]
            
            # SIMPLE CLASS CHECK: Is this class in our alert_classes?
            if self._alert_classes is not None:
                if class_label not in self._alert_classes:
                    continue
            
            # Check line intersection
            angle_of_intersection = self._line.find_angle_of_intersection(track)
            
            if angle_of_intersection is None:
                continue
            
            # Adjust angle based on line orientation
            if self._line.start_position[0] > self._line.end_position[0]:
                angle_of_intersection = abs(angle_of_intersection)
                angle_of_intersection = 180 - angle_of_intersection
                angle_of_intersection *= -1
                
                if self._line.start_position[1] < self._line.end_position[1]:
                    angle_of_intersection = angle_of_intersection * -1
            
            if angle_of_intersection > 180:
                angle_of_intersection = 180 - angle_of_intersection
            
            # Check direction filter
            if not self._matches_direction(angle_of_intersection):
                continue
            
            # Get bbox for size check
            bbox = track.get_state()[0]
            
            # Check object size filter
            if not self._is_valid_object_size(bbox):
                continue
            
            # All checks passed - generate event
            line_intrusion_direction = 'left' if angle_of_intersection > 0 else 'right'
            
            # Draw red bbox
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Register notification
            self._register_notification(track_id, frame, line_intrusion_direction, bbox)
            track.notification_generated = True
            
            logger.warning(
                f"[LineIntrusion:{self._stream_id}:{self._instance_id}] "
                f"Track {track_id} ({class_label}) crossed {line_intrusion_direction}"
            )

    def _register_notification(self, track_id: int, frame, direction: str, bbox):
        """Register notification - uses async or sync based on flag"""
        if self._use_async and self._shared_async_handler is not None:
            # Async mode
            metadata = {
                "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
                "direction": direction,
                "feature_type": "line_intrusion",
            }
            
            try:
                asyncio.create_task(
                    self._shared_async_handler.notify_event(
                        stream_id=self._stream_id,
                        object_id=track_id,
                        instance_id=self._instance_id,
                        alert_type="Line Intrusion",
                        frame=frame,
                        direction=direction,
                        confidence=1.0,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to queue async notification: {e}")
                self._register_notification_sync(track_id, frame, direction)
        else:
            # Sync mode (default)
            self._register_notification_sync(track_id, frame, direction)

    def _register_notification_sync(self, track_id: int, frame, direction: str):
        """Sync notification - saves locally"""
        notification = Notification(
            object_id=track_id,
            frame=frame,
            instance_id=self._instance_id,
            alert_type="Line Intrusion",
            stream_id=self._stream_id,
            line_intrusion_notif_direction=direction
        )
        notification.register()

    @property
    def line_points(self):
        return [self._line.start_position, self._line.end_position]
