import threading
import cv2
import time
import numpy as np
from typing import List, Dict, Any
from src.models.tracked_object import TrackedObject


class LiveVisualization:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_objects: List[TrackedObject] = []
        self.latest_intrusions: List[Dict[str, Any]] = []
        self.latest_geometries: List[Dict[str, Any]] = []

    def update_frame_visualization(self, frame, tracked_objects: List[TrackedObject], intrusions: List[Dict[str, Any]],
                     geometries: List[Dict[str, Any]] = None):
        """Store latest frame + overlay data (thread-safe)."""
        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_objects = list(tracked_objects)
            self.latest_intrusions = list(intrusions)
            self.latest_geometries = list(geometries) if geometries else []

    def start_display(self):
        self.running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    objects = list(self.latest_objects)
                    intrusions = list(self.latest_intrusions)
                    geometries = list(self.latest_geometries)
                else:
                    objects = []
                    intrusions = []
                    geometries = []

            if frame is not None:
                # Draw dynamic geometries (lines, zones, etc.)
                self._draw_geometries(frame, geometries)

                # Draw tracked objects
                for obj in objects:
                    x1, y1, x2, y2 = map(int, obj.bbox)
                    label = f"{obj.class_name}#{obj.track_id} {obj.confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw intrusion alerts
                for intrusion_event in intrusions:
                    bbox = list(map(int, intrusion_event['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                    if "position" in intrusion_event:
                        position = intrusion_event['position']
                        cv2.putText(frame, 'INTRUSION',
                                    (int(position[0]) - 30, int(position[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                break
            time.sleep(0.01)

        cv2.destroyWindow(self.window_name)

    def _draw_geometries(self, frame, geometries: List[Dict[str, Any]]):
        """Draw all geometric elements (lines, zones, etc.) on the frame."""
        for geometry in geometries:
            geometry_type = geometry.get("type", "")

            if geometry_type == "line":
                line_points = geometry.get("points", [])
                if len(line_points) >= 2:
                    cv2.line(frame, tuple(line_points[0]), tuple(line_points[1]), (0, 255, 255), 3)

            elif geometry_type == "zone":
                zone_points = geometry.get("points", [])
                if len(zone_points) >= 3:  # Need at least 3 points for a polygon
                    points_array = np.array(zone_points, np.int32)
                    cv2.polylines(frame, [points_array], isClosed=True, color=(255, 0, 0), thickness=2)
                    # Optional: fill with semi-transparent color
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points_array], (255, 0, 0))
                    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def stop(self):
        self.running = False