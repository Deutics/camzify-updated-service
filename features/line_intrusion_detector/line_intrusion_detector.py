"""Line Intrusion Detector - Pure classifier (no model loading)"""

import cv2
from typing import List, Dict
from Utils.Line.Line import Line
from notification.notification_handler import Notification
from utils_main.logger import get_logger

logger = get_logger(__name__)


class LineIntrusionDetector:
    """Line intrusion classifier - checks if tracked objects cross line"""
    
    def __init__(self, 
                 line_coords,
                 instance_id,
                 stream_id,
                 direction_to_check=None):
        """
        Initialize Line Intrusion Detector (NO MODEL LOADING)
        
        Args:
            line_coords: [(x1, y1), (x2, y2)] - Line start and end points
            instance_id: Unique identifier for this detector instance
            stream_id: Stream identifier this detector is attached to
            direction_to_check: Dict with 'left' and 'right' booleans (None = both directions)
        """
        if direction_to_check is None:
            direction_to_check = {"left": True, "right": True}
        
        self._instance_id = instance_id
        self._stream_id = stream_id
        self._direction_to_check = direction_to_check
        
        # Initialize line (geometric only)
        self._line = Line(start_position=line_coords[0], end_position=line_coords[1])
        
        # logger.info(f"[LineIntrusion:{stream_id}:{instance_id}] Initialized (no model)")
        # logger.info(f"  Line: {line_coords[0]} -> {line_coords[1]}")
        # logger.info(f"  Direction: {direction_to_check}")

    def process(self, tracked_objects: List, tracks: List, frame) -> List[Dict]:
        """
        Check if tracked objects cross the line (CLASSIFICATION ONLY)
        
        Args:
            tracked_objects: List of tracked objects from detector
            tracks: SORT tracks with position history
            frame: Frame for visualization and notifications
            
        Returns:
            List of intrusion event dicts
        """
        events = []
        
        # Draw line on frame
        cv2.line(frame, self._line.start_position, self._line.end_position, (0, 0, 255), 2)
        
        # Check each track for line crossing
        for track in tracks:
            # Find angle of intersection (returns None if no intersection)
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
            
            # Check if direction matches configuration
            if (angle_of_intersection > 0 and self._direction_to_check["left"]) or \
               (angle_of_intersection < 0 and self._direction_to_check["right"]):
                
                # Determine direction
                line_intrusion_direction = 'left' if angle_of_intersection > 0 else 'right'
                
                # Get track info
                bbox = track.get_state()[0]
                track_id = track.track_id + 1  # SORT uses 0-indexed
                
                # Draw red bbox on intruded track
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Create event
                event = {
                    'stream_id': self._stream_id,
                    'instance_id': self._instance_id,
                    'track_id': track_id,
                    'direction': line_intrusion_direction,
                    'bbox': bbox.tolist(),
                    'angle': angle_of_intersection
                }
                events.append(event)
                
                # Register notification
                self._register_notification(track_id, frame, line_intrusion_direction)
                
                # Mark as notified to prevent duplicates
                track.notification_generated = True
                
                logger.warning(f"[LineIntrusion:{self._stream_id}:{self._instance_id}] "
                             f"Track {track_id} crossed {line_intrusion_direction}")
        
        return events

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
        """Get line coordinates for visualization"""
        return [self._line.start_position, self._line.end_position]
