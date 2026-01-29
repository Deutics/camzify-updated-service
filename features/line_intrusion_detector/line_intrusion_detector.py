"""Line Intrusion Detector with time bounds, object size, and direction checks"""

import cv2
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from Utils.Line.Line import Line
from notification.notification_handler import Notification
from utils_main.logger import get_logger

logger = get_logger(__name__)


class LineIntrusionDetector:
    """Line intrusion classifier - checks if tracked objects cross line"""
    
    def __init__(self, 
                 line_coords: List[Tuple[int, int]],
                 instance_id: str,
                 stream_id: str,
                 direction_to_use: int = 0,
                 bounding_box_start: Optional[Tuple[int, int]] = None,
                 bounding_box_end: Optional[Tuple[int, int]] = None,
                 precision_factor: int = 0,
                 time_bound_start: str = "00:00:00",
                 time_bound_end: str = "23:59:59",
                 alert_classes: Optional[List[str]] = None):
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
            alert_classes: List of object classes to detect
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
        
        # Alert classes (filter by object type)
        self._alert_classes = set(alert_classes) if alert_classes else None
        
        logger.info(f"[LineIntrusion:{stream_id}:{instance_id}] Initialized")
        logger.info(f"  Direction: {direction_to_use} (0=any, 1=left, -1=right)")
        logger.info(f"  Time: {time_bound_start} - {time_bound_end}")
        logger.info(f"  Size filter: {self._size_thresholds is not None}")
        logger.info(f"  Alert classes: {alert_classes}")

    def _calculate_size_thresholds(self, bbox_start, bbox_end, precision_factor):
        """Calculate min/max size thresholds from bounding box and precision factor"""
        if not bbox_start or not bbox_end or precision_factor == 0:
            return None
        
        # Calculate reference object size
        ref_width = abs(bbox_end[0] - bbox_start[0])
        ref_height = abs(bbox_end[1] - bbox_start[1])
        
        if ref_width == 0 or ref_height == 0:
            return None
        
        # Precision factor is percentage (e.g., 20 means ±20%)
        precision = precision_factor * 0.01
        
        min_width = ref_width - (ref_width * precision)
        max_width = ref_width + (ref_width * precision)
        min_height = ref_height - (ref_height * precision)
        max_height = ref_height + (ref_height * precision)
        
        logger.debug(f"Size thresholds: W[{min_width:.1f}-{max_width:.1f}] H[{min_height:.1f}-{max_height:.1f}]")
        
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
            return True  # Default to allowing if parsing fails

    def _is_valid_object_size(self, bbox: List[float]) -> bool:
        """Check if detected object size is within configured thresholds"""
        if self._size_thresholds is None:
            return True  # No size filtering
        
        obj_width = bbox[2] - bbox[0]
        obj_height = bbox[3] - bbox[1]
        
        return (
            self._size_thresholds["min_width"] <= obj_width <= self._size_thresholds["max_width"] and
            self._size_thresholds["min_height"] <= obj_height <= self._size_thresholds["max_height"]
        )

    def _matches_direction(self, angle: float) -> bool:
        """Check if crossing direction matches configuration"""
        if self._direction_to_use == 0:
            return True  # Any direction
        
        is_left = angle > 0
        
        if self._direction_to_use == 1:
            return is_left  # Only left
        else:  # direction_to_use == 2
            return not is_left  # Only right

    def process(self, tracks: List, frame):
        """Check for line intrusions with all filters applied"""
        events = []
        
        # Draw line
        cv2.line(frame, self._line.start_position, self._line.end_position, (0, 0, 255), 2)
        
        # Check 1: Time bounds
        if not self._is_within_time_bounds():
            return events
        
        # Check each track
        for track in tracks:
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
            
            # Check 2: Direction filter
            if not self._matches_direction(angle_of_intersection):
                continue
            
            # Get bbox for size check
            bbox = track.get_state()[0]
            
            # Check 3: Object size filter
            if not self._is_valid_object_size(bbox):
                continue
            
            # All checks passed - generate event
            track_id = track.track_id + 1
            line_intrusion_direction = 'left' if angle_of_intersection > 0 else 'right'
            
            # Draw red bbox
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # event = {
            #     'stream_id': self._stream_id,
            #     'instance_id': self._instance_id,
            #     'track_id': track_id,
            #     'direction': line_intrusion_direction,
            #     'bbox': bbox.tolist(),
            #     'angle': angle_of_intersection
            # }
            # events.append(event)
            
            self._register_notification(track_id, frame, line_intrusion_direction)
            track.notification_generated = True
            
            logger.warning(
                f"[LineIntrusion:{self._stream_id}:{self._instance_id}] "
                f"Track {track_id} crossed {line_intrusion_direction}"
            )
        
        # return events

    def _register_notification(self, track_id, frame, direction):
        """Save notification snapshot"""
        notification = Notification(
            object_id=track_id,
            frame=frame,
            instance_id=self._instance_id,
            stream_id=self._stream_id,
            line_intrusion_notif_direction=direction
        )
        notification.register()

    @property
    def line_points(self):
        return [self._line.start_position, self._line.end_position]
