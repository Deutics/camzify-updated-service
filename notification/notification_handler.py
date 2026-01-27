"""Simplified Notification Module - Saves intrusion snapshots locally"""

import cv2
import os
from datetime import datetime
from pathlib import Path
import secrets


class Notification:
    def __init__(self, object_id, frame, instance_id, stream_id=None,
                 line_intrusion_notif_direction=None, output_dir="output/intrusions"):
        """
        Initialize notification
        
        Args:
            object_id: Track ID of the intruding object
            frame: Frame image (numpy array)
            instance_id: Instance identifier (feature instance)
            stream_id: Stream identifier
            line_intrusion_notif_direction: 'left' or 'right'
            output_dir: Directory to save snapshots
        """
        self._event_time = datetime.now()
        self._object_id = str(object_id)
        self._frame = frame
        self._instance_id = instance_id
        self._stream_id = stream_id or "unknown"
        self._line_intrusion_notif_direction = line_intrusion_notif_direction
        self._output_dir = output_dir
        
        # Create output directory
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

    def _generate_filename(self):
        """Generate unique filename for snapshot"""
        timestamp = self._event_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        filename = f"{self._stream_id}_obj{self._object_id}_{timestamp}.jpg"
        return os.path.join(self._output_dir, filename)

    def register(self):
        """Save notification snapshot to disk"""
        try:
            filepath = self._generate_filename()
            
            # Save frame as JPEG
            success = cv2.imwrite(filepath, self._frame)
            
            if success:
                print(f"[NOTIFICATION] Saved: {filepath}")
                print(f"              Stream: {self._stream_id}")
                print(f"              Object ID: {self._object_id}")
                print(f"              Direction: {self._line_intrusion_notif_direction}")
                print(f"              Instance: {self._instance_id}")
                return filepath
            else:
                print(f"[NOTIFICATION ERROR] Failed to save: {filepath}")
                return None
                
        except Exception as e:
            print(f"[NOTIFICATION ERROR] Exception: {e}")
            return None
