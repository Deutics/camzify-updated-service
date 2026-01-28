from typing import Tuple, List, Dict


class CoordinateMapper:
    def __init__(self, width: int, height: int):
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    def to_pixels(self, x_norm: float, y_norm: float) -> Tuple[int, int]:
        x = max(0.0, min(1.0, float(x_norm or 0.0)))
        y = max(0.0, min(1.0, float(y_norm or 0.0)))
        return int(x * self.width), int(y * self.height)

    def build_line_points(self, line_points_norm: List[Tuple[float, float]]) -> List[Tuple

[int, int]]:
        if not line_points_norm or len(line_points_norm) < 2:
            line_points_norm = [(0.0, 0.0), (1.0, 1.0)]
        return [self.to_pixels(x, y) for x, y in line_points_norm[:2]]

    def build_bbox_points(self, start: Tuple[float, float], end: Tuple[float, float]) -> Dict[str, Tuple[int, int]]:
        return {
            "bbox_start": self.to_pixels(*start),
            "bbox_end": self.to_pixels(*end)
        }