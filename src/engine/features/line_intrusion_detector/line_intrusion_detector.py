from typing import List, Tuple, Dict, Optional
from src.data_models.tracked_object import TrackedObject
from src.data_models.frame_data import FrameData
from src.engine.feature_base import FeatureBase
from src.utils.logger import get_logger
from src.engine.features.feature_utils import is_point_inside_bbox, CooldownTracker, has_line_crossed, direction_matches

logger = get_logger(__name__)


class LineIntrusionDetector(FeatureBase):
    """Line intrusion detector with bounding box, cooldown, and direction support."""

    def __init__(self, config: Dict, stream_id: str, alert_system=None, visualizer=None):
        self.stream_id = stream_id
        self.alert_system = alert_system
        self.visualizer = visualizer

        self._config = config or {}
        self.instance_id: Optional[str] = self._config.get("instance_id")

        self.line_points: List[Tuple[int, int]] = self._config.get("line_points", [])
        self.alert_classes = set(self._config.get("alert_classes", [])) if self._config.get("alert_classes") else None
        self.cooldown_duration: float = float(self._config.get("cooldown", 2.0))

        self.bbox_start: Tuple[int, int] = self._config.get("bounding_box_start", (0, 0))
        self.bbox_end: Tuple[int, int] = self._config.get("bounding_box_end", (640, 480))
        self.precision_factor: float = float(self._config.get("precision_factor", 1.0))

        self.direction_to_use: int = int(self._config.get("direction_to_use", 0))

        self._compute_size_thresholds()

        self.cooldowns = CooldownTracker(self.cooldown_duration)

    def _compute_size_thresholds(self):
        """Compute expected object size thresholds from bounding box dimensions."""
        reference_width = abs(self.bbox_end[0] - self.bbox_start[0])
        reference_height = abs(self.bbox_end[1] - self.bbox_start[1])

        if reference_width > 0 and reference_height > 0 and self.precision_factor > 0:
            pf = self.precision_factor * 0.01
            self.min_obj_width = reference_width - (reference_width * pf)
            self.max_obj_width = reference_width + (reference_width * pf)
            self.min_obj_height = reference_height - (reference_height * pf)
            self.max_obj_height = reference_height + (reference_height * pf)
            logger.debug(
                f"[ILD:{self.stream_id}:{self.instance_id}] Size thresholds: "
                f"width={self.min_obj_width:.1f}-{self.max_obj_width:.1f}, "
                f"height={self.min_obj_height:.1f}-{self.max_obj_height:.1f}"
            )
        else:
            self.min_obj_width = None
            self.max_obj_width = None
            self.min_obj_height = None
            self.max_obj_height = None
            logger.debug(f"[ILD:{self.stream_id}:{self.instance_id}] No size filtering (invalid bbox or precision)")

    def check_intrusion(self, tracked_objects: List[TrackedObject],
                        frame_data: Optional[FrameData], meta: Dict) -> List[Dict]:
        events = []

        for obj in tracked_objects:
            if self.alert_classes and obj.class_name not in self.alert_classes:
                continue

            obj_center = obj.center()

            if not is_point_inside_bbox(obj_center, self.bbox_start, self.bbox_end):
                continue

            if self.min_obj_width is not None:
                obj_width = obj.bbox[2] - obj.bbox[0]
                obj_height = obj.bbox[3] - obj.bbox[1]

                if not (self.min_obj_width <= obj_width <= self.max_obj_width and
                        self.min_obj_height <= obj_height <= self.max_obj_height):
                    continue

            if len(obj.history) < 2:
                continue

            object_uid = f"{obj.stream_id}_{obj.track_id}"
            max_segments = min(len(obj.history), 6)

            for i in range(1, max_segments):
                p1 = obj.history[-i]
                p0 = obj.history[-(i + 1)]

                if has_line_crossed(p0, p1, self.line_points[0], self.line_points[1]):
                    now = obj.timestamp
                    if not self.cooldowns.is_in_cooldown(object_uid, now):
                        if direction_matches(p0, p1, self.direction_to_use):
                            event = {
                                'stream_id': obj.stream_id,
                                'track_id': obj.track_id,
                                'class_name': obj.class_name,
                                'bbox': obj.bbox,
                                'position': obj_center,
                                'timestamp': obj.timestamp,
                                'rule_type': 'line_intrusion',
                                'direction': 'any' if self.direction_to_use == 0 else ('left' if p1[0] < p0[0] else 'right'),
                                'instance_id': self.instance_id
                            }
                            events.append(event)
                            self.cooldowns.set(object_uid, now)
                    break

        return events

    def update_config(self, new_cfg: Dict) -> None:
        """Update feature configuration at runtime."""
        if "line_points" in new_cfg:
            self.line_points = new_cfg["line_points"]
            logger.info(f"[ILD:{self.stream_id}] update_config line_points -> {self.line_points}")
        if "alert_classes" in new_cfg:
            self.alert_classes = set(new_cfg["alert_classes"])
        if "cooldown" in new_cfg:
            self.cooldown_duration = float(new_cfg["cooldown"])
            self.cooldowns.duration = self.cooldown_duration

        size_changed = False
        if "bounding_box_start" in new_cfg:
            self.bbox_start = new_cfg["bounding_box_start"]
            size_changed = True
        if "bounding_box_end" in new_cfg:
            self.bbox_end = new_cfg["bounding_box_end"]
            size_changed = True
        if "precision_factor" in new_cfg:
            self.precision_factor = float(new_cfg["precision_factor"])
            size_changed = True

        if size_changed:
            self._compute_size_thresholds()

        if "direction_to_use" in new_cfg:
            self.direction_to_use = int(new_cfg["direction_to_use"])

    def shutdown(self) -> None:
        """Release resources."""
        self.cooldowns._timestamps.clear()
        logger.info(f"[ILD:{self.stream_id}] LineIntrusionDetector shutdown")