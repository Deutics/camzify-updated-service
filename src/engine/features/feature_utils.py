from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger("CamzifyService.features.utils")


def is_point_inside_bbox(point: Tuple[float, float], bbox_start: Tuple[int, int], bbox_end: Tuple[int, int]) -> bool:
    """Check if a point (x, y) is inside a bounding box."""
    try:
        if not isinstance(point, (tuple, list)) or len(point) != 2 or not all(isinstance(v, (int, float)) for v in point):
            logger.debug(f"[ROI] Invalid point: {point}")
            return False

        x, y = point
        x1, y1 = bbox_start
        x2, y2 = bbox_end

        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
            logger.debug(f"[ROI] Invalid bbox: start={bbox_start}, end={bbox_end}")
            return False

        return x1 <= x <= x2 and y1 <= y <= y2

    except Exception as e:
        logger.exception(f"[ROI] Error checking point inside bbox: {e}")
        return False


def has_line_crossed(p0: Tuple[float, float], p1: Tuple[float, float],
                     line_start: Tuple[int, int], line_end: Tuple[int, int]) -> bool:
    """Check if the line segment p0->p1 crosses line_start->line_end."""
    try:
        def side(pt):
            return ((line_end[0] - line_start[0]) * (pt[1] - line_start[1]) -
                    (line_end[1] - line_start[1]) * (pt[0] - line_start[0]))
        return side(p0) * side(p1) < 0
    except Exception as e:
        logger.exception(f"[LINE] Error in has_line_crossed: {e}")
        return False


class CooldownTracker:
    """Manages cooldown per object ID."""
    def __init__(self, duration_s: float):
        self.duration = duration_s
        self._timestamps: Dict[str, float] = {}

    def is_in_cooldown(self, key: str, now: float) -> bool:
        last = self._timestamps.get(key)
        return (now - last) < self.duration if last is not None else False

    def set(self, key: str, now: float):
        self._timestamps[key] = now


def direction_matches(p0: Tuple[float, float], p1: Tuple[float, float], direction_to_use: int) -> bool:
    """Check if movement direction matches configuration."""
    if direction_to_use == 0:
        return True  # Any direction allowed

    direction = 'left' if p1[0] < p0[0] else 'right'
    return (direction_to_use == 1 and direction == 'left') or (direction_to_use == 2 and direction == 'right')


def get_overlay_geometry(feature_instances: List) -> List[Dict]:
    """
    Extract geometry for visualizer overlays from feature instances.
    Returns list of dicts with 'type' and 'points'.
    """
    overlay = []
    for inst in feature_instances:
        if hasattr(inst, "line_points") and inst.line_points:
            overlay.append({"type": "line", "points": inst.line_points})
        if hasattr(inst, "zone_points") and inst.zone_points:
            overlay.append({"type": "zone", "points": inst.zone_points})
    return overlay
