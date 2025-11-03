# simple_tracker.py
import time
from typing import List, Dict, Optional, Tuple
from data_models.TrackedObject import TrackedObject


def calculate_intersection_over_union(bounding_box_a, bounding_box_b) -> float:
    """Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    ax1, ay1, ax2, ay2 = bounding_box_a
    bx1, by1, bx2, by2 = bounding_box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    iw = max(0.0, intersection_x2 - intersection_x1)
    ih = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - intersection_area
    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


def _center(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


class TrackedObjectState:
    """Represents the state of a single tracked object across frames."""

    def __init__(self, track_id, bounding_box, class_name, timestamp, confidence=1.0, class_id=-1, max_history=60):
        self.track_id = track_id
        self.current_bounding_box = bounding_box  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.class_id = class_id
        self.last_seen_timestamp = timestamp
        self.confidence = confidence

        # Motion tracking
        self.previous_bounding_box = None
        self.previous_timestamp = None

        # History of last N center positions with timestamps
        self.position_history = []  # [(center_x, center_y, timestamp), ...]
        self.max_history = max_history
        self._update_position_history(bounding_box, timestamp)

        # Track quality metrics
        self.successful_matches = 1
        self.consecutive_misses = 0

        # Class change hysteresis to avoid flapping
        self._class_stability = 0

    def _update_position_history(self, bbox, timestamp):
        """Update the position history with new center point."""
        cx, cy = _center(bbox)
        self.position_history.append((cx, cy, timestamp))
        if len(self.position_history) > self.max_history:
            self.position_history = self.position_history[-self.max_history:]

    def update_with_detection(self, bbox, timestamp, confidence=1.0, class_id=-1, class_name: Optional[str] = None, class_change_hysteresis: int = 2):
        """Update track state with new detection."""
        self.previous_bounding_box = self.current_bounding_box
        self.previous_timestamp = self.last_seen_timestamp

        self.current_bounding_box = bbox
        self.last_seen_timestamp = timestamp
        self.confidence = confidence

        # Update class with hysteresis (avoid flip-flop on noisy class predictions)
        if class_id != -1 and class_id != self.class_id:
            self._class_stability += 1
            if self._class_stability >= class_change_hysteresis:
                self.class_id = class_id
                if class_name:
                    self.class_name = class_name
                self._class_stability = 0
        else:
            # Same class (or unknown): reset stability and update name if provided
            self._class_stability = 0
            if class_name:
                self.class_name = class_name

        self._update_position_history(bbox, timestamp)
        self.successful_matches += 1
        self.consecutive_misses = 0

    def predict_next_position(self, current_timestamp):
        """Predict where this object will be at current_timestamp based on motion."""
        # If no previous position, return current position
        if self.previous_bounding_box is None or self.previous_timestamp is None:
            return self.current_bounding_box

        # Calculate time deltas
        time_since_last_update = max(1e-3, current_timestamp - self.last_seen_timestamp)
        time_between_updates = max(1e-3, self.last_seen_timestamp - self.previous_timestamp)

        # Calculate center points and velocity
        pcx, pcy = _center(self.previous_bounding_box)
        ccx, ccy = _center(self.current_bounding_box)

        vx = (ccx - pcx) / time_between_updates
        vy = (ccy - pcy) / time_between_updates

        # Optional: clamp velocity to avoid exploding predictions on timestamp glitches
        # max_pix_per_sec = 2000.0
        # vmag = max(1e-6, (vx*vx + vy*vy) ** 0.5)
        # if vmag > max_pix_per_sec:
        #     scale = max_pix_per_sec / vmag
        #     vx *= scale; vy *= scale

        # Predict new center position
        pred_cx = ccx + vx * time_since_last_update
        pred_cy = ccy + vy * time_since_last_update

        # Keep same width and height
        w = self.current_bounding_box[2] - self.current_bounding_box[0]
        h = self.current_bounding_box[3] - self.current_bounding_box[1]

        # Reconstruct bounding box around predicted center
        predicted_box = [
            pred_cx - w * 0.5,
            pred_cy - h * 0.5,
            pred_cx + w * 0.5,
            pred_cy + h * 0.5
        ]
        return predicted_box

    def get_center_history(self):
        """Get list of (x, y) center positions from history."""
        return [(x, y) for x, y, _ in self.position_history]

    def get_full_history(self):
        """Get full history with timestamps: [(x, y, timestamp), ...]"""
        return list(self.position_history)


class SimpleTracker:
    """Simple object tracker using IoU matching with motion prediction."""

    def __init__(self, iou_matching_threshold=0.3, max_object_age_seconds=5.0, max_center_dist: float = 200.0):
        """
        max_center_dist: optional center-distance gate (pixels) used alongside IoU to reduce ID switches.
        """
        self.iou_matching_threshold = float(iou_matching_threshold)
        self.max_object_age_seconds = float(max_object_age_seconds)
        self.max_center_dist2 = float(max_center_dist) * float(max_center_dist)

        self._active_tracks: Dict[int, TrackedObjectState] = {}  # track_id -> TrackedObjectState
        self._next_available_id = 1

    def _generate_new_track_id(self) -> int:
        """Generate a unique track ID."""
        track_id = self._next_available_id
        self._next_available_id += 1
        if self._next_available_id > 1_000_000_000:
            self._next_available_id = 1
        return track_id

    def update_tracker(self, new_detections: List[Dict], frame_timestamp: float, stream_id: str, frame_number: int) -> List[TrackedObject]:
        """
        Update tracker with new detections and return TrackedObject instances.

        Args:
            new_detections: List of detection dicts with 'bbox', 'class_name', etc.
            frame_timestamp: Timestamp of current frame
            stream_id: ID of the video stream
            frame_number: Frame number in the stream

        Returns:
            List of TrackedObject instances with track IDs assigned
        """
        current_timestamp = frame_timestamp if frame_timestamp is not None else time.time()
        tracked_objects: List[TrackedObject] = []

        # Normalize and filter detections lacking bbox
        if not new_detections:
            self._remove_stale_tracks(current_timestamp)
            return []

        detections = [d for d in new_detections if isinstance(d.get('bbox', None), (list, tuple)) and len(d['bbox']) == 4]
        if not detections:
            self._remove_stale_tracks(current_timestamp)
            return []

        # Prepare existing tracks and their predicted positions at current time
        existing_tracks = list(self._active_tracks.values())
        predicted_positions = [t.predict_next_position(current_timestamp) for t in existing_tracks]

        # Build potential matches (track_idx, det_idx, score)
        potential_matches = []
        for t_idx, track in enumerate(existing_tracks):
            pred_box = predicted_positions[t_idx]
            pcx, pcy = _center(pred_box)
            for d_idx, det in enumerate(detections):
                det_box = det['bbox']
                dcx, dcy = _center(det_box)

                # Center distance gate to avoid wild matches
                dx = pcx - dcx
                dy = pcy - dcy
                if (dx * dx + dy * dy) > self.max_center_dist2:
                    continue

                # IoU gate
                iou_score = calculate_intersection_over_union(pred_box, det_box)
                if iou_score >= self.iou_matching_threshold:
                    potential_matches.append((t_idx, d_idx, iou_score))

        # Sort matches by IoU score (best matches first)
        potential_matches.sort(key=lambda match: match[2], reverse=True)

        # Assign detections to tracks (greedy matching)
        matched_track_ids = set()
        matched_detection_indices = set()

        for t_idx, d_idx, _ in potential_matches:
            track = existing_tracks[t_idx]
            if track.track_id in matched_track_ids or d_idx in matched_detection_indices:
                continue

            det = detections[d_idx]
            track.update_with_detection(
                bbox=det['bbox'],
                timestamp=current_timestamp,
                confidence=det.get('confidence', 1.0),
                class_id=det.get('class_id', -1),
                class_name=det.get('class_name')
            )

            matched_track_ids.add(track.track_id)
            matched_detection_indices.add(d_idx)

            tracked_objects.append(
                TrackedObject(
                    track_id=track.track_id,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    bbox=track.current_bounding_box,
                    confidence=track.confidence,
                    stream_id=stream_id,
                    timestamp=current_timestamp,
                    frame_number=frame_number,
                    history=track.get_center_history()
                )
            )

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx in matched_detection_indices:
                continue

            new_track_id = self._generate_new_track_id()
            new_track = TrackedObjectState(
                track_id=new_track_id,
                bounding_box=det['bbox'],
                class_name=det.get('class_name', 'unknown'),
                timestamp=current_timestamp,
                confidence=det.get('confidence', 1.0),
                class_id=det.get('class_id', -1)
            )
            self._active_tracks[new_track_id] = new_track

            tracked_objects.append(
                TrackedObject(
                    track_id=new_track_id,
                    class_id=new_track.class_id,
                    class_name=new_track.class_name,
                    bbox=new_track.current_bounding_box,
                    confidence=new_track.confidence,
                    stream_id=stream_id,
                    timestamp=current_timestamp,
                    frame_number=frame_number,
                    history=new_track.get_center_history()
                )
            )

        # Update missed counts and remove stale tracks
        self._update_unmatched_tracks(matched_track_ids)
        self._remove_stale_tracks(current_timestamp)

        return tracked_objects

    def _update_unmatched_tracks(self, matched_track_ids):
        """Update consecutive miss counts for tracks that weren't matched."""
        for track_id, track in self._active_tracks.items():
            if track_id not in matched_track_ids:
                track.consecutive_misses += 1

    def _remove_stale_tracks(self, current_timestamp):
        """Remove tracks that haven't been seen for too long."""
        stale_track_ids = []
        for track_id, track in self._active_tracks.items():
            if (current_timestamp - track.last_seen_timestamp) > self.max_object_age_seconds:
                stale_track_ids.append(track_id)
        for tid in stale_track_ids:
            del self._active_tracks[tid]

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks."""
        return len(self._active_tracks)

    def get_track_by_id(self, track_id) -> Optional[TrackedObjectState]:
        """Get TrackedObjectState by track ID (for debugging)."""
        return self._active_tracks.get(track_id)