# TrackedObjectManager.py
from typing import Dict, List, Tuple, Any, Optional
from src.services.tracking.simple_tracker import SimpleTracker

class TrackedObjectManager:
    def __init__(self, latest_only: bool = True):
        """
        latest_only:
          - True: return only the latest frame's tracked objects per stream (recommended for real-time rule engines).
          - False: return tracked objects for all frames received in this batch per stream.
        """
        self._trackers: Dict[str, SimpleTracker] = {}  # stream_id -> SimpleTracker()
        self._latest_only = bool(latest_only)

        # Optional per-stream tracker params you can set at runtime if needed.
        # Example: self._tracker_overrides["cam-1"] = {"max_object_age_seconds": 5.0}
        self._tracker_overrides: Dict[str, Dict[str, Any]] = {}

    def _get_tracker(self, stream_id: str) -> SimpleTracker:
        """Get or create a SimpleTracker for the given stream."""
        if stream_id not in self._trackers:
            tracker = SimpleTracker()
            # Apply any per-stream overrides (if previously set)
            overrides = self._tracker_overrides.get(stream_id)
            if overrides:
                self._apply_overrides(tracker, overrides)
            self._trackers[stream_id] = tracker
        return self._trackers[stream_id]

    def set_tracker_overrides(self, stream_id: str, overrides: Dict[str, Any]) -> None:
        """
        Optional: Adjust tracker parameters for a specific stream.
        Call before frames arrive to ensure new tracker uses these params.
        Allowed keys depend on SimpleTracker __init__ signature.
        """
        self._tracker_overrides[stream_id] = dict(overrides)
        if stream_id in self._trackers:
            self._apply_overrides(self._trackers[stream_id], overrides)

    def _apply_overrides(self, tracker: SimpleTracker, overrides: Dict[str, Any]) -> None:
        """
        Apply supported overrides on an existing tracker instance.
        Safe: only touches known attributes if present.
        """
        for k, v in overrides.items():
            if hasattr(tracker, k):
                setattr(tracker, k, v)

    def update_batch(self, detections_by_sid: Dict[str, List[Tuple[Any, List[Dict]]]]) -> Dict[str, List[Any]]:
        """
        Input:  { stream_id -> [(FrameData, [raw_detections]), ...] }
        Output: { stream_id -> [TrackedObject, ...] }
        Behavior:
          - Ensures chronological processing per stream by sorting on frame_number.
          - If latest_only=True: returns only the latest frame's tracked objects per stream.
          - Else: returns concatenated tracked objects for all frames in this batch per stream.
        """
        tracked_results: Dict[str, List[Any]] = {}

        for stream_id, frames_data_detection in detections_by_sid.items():
            tracker = self._get_tracker(stream_id)
            if not frames_data_detection:
                tracked_results[stream_id] = []
                continue

            # Sort by frame_number to guarantee temporal consistency for the tracker
            try:
                frames_data_detection = sorted(
                    frames_data_detection, key=lambda pair: getattr(pair[0], "frame_number", 0)
                )
            except Exception:
                # If frame_number not present or sortable, proceed as-is
                pass

            if self._latest_only:
                # Process all frames but only return the last frame's tracked objects
                last_tracked = None
                for frame_data, detections in frames_data_detection:
                    last_tracked = tracker.update_tracker(
                        new_detections=detections,
                        frame_timestamp=getattr(frame_data, "timestamp", None),
                        stream_id=stream_id,
                        frame_number=getattr(frame_data, "frame_number", 0),
                    )
                tracked_results[stream_id] = last_tracked or []
            else:
                # Concatenate tracked objects for all frames in this batch (could trigger duplicate rule evals)
                per_stream_tracked: List[Any] = []
                for frame_data, detections in frames_data_detection:
                    tracked_objects = tracker.update_tracker(
                        new_detections=detections,
                        frame_timestamp=getattr(frame_data, "timestamp", None),
                        stream_id=stream_id,
                        frame_number=getattr(frame_data, "frame_number", 0),
                    )
                    per_stream_tracked.extend(tracked_objects)
                tracked_results[stream_id] = per_stream_tracked

        return tracked_results

    def get_tracker_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about active trackers (for debugging/monitoring)."""
        stats = {}
        for stream_id, tracker in self._trackers.items():
            stats[stream_id] = {
                'active_tracks': tracker.get_active_track_count()
            }
        return stats