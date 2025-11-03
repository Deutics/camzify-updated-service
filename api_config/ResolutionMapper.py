class ResolutionMapper:
    """
    Small helper to convert normalized coordinates to pixel coordinates.
    """

    def __init__(self, width: int, height: int):
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    def to_pixels(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        x = float(x_norm or 0.0)
        y = float(y_norm or 0.0)
        # clamp to [0,1] for safety
        if x < 0.0: x = 0.0
        if x > 1.0: x = 1.0
        if y < 0.0: y = 0.0
        if y > 1.0: y = 1.0
        return int(x * self.width), int(y * self.height)

    def build_line_points(self, cfg: dict) -> list:
        # cfg expected to have "line_points_norm": [(x1,y1),(x2,y2)]
        pts = cfg.get("line_points_norm")
        if not pts or not isinstance(pts, list) or len(pts) < 2:
            pts = [(0.0, 0.0), (1.0, 1.0)]
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return [self.to_pixels(x1, y1), self.to_pixels(x2, y2)]

    def build_bbox_points(self, cfg: dict) -> dict:
        # cfg expected to have bounding_box_start/end normalized tuples
        start = cfg.get("bounding_box_start") or cfg.get("bounding_box_start", (0.0, 0.0))
        end = cfg.get("bounding_box_end") or cfg.get("bounding_box_end", (1.0, 1.0))
        sx, sy = start
        ex, ey = end
        return {"bbox_start": self.to_pixels(sx, sy), "bbox_end": self.to_pixels(ex, ey)}


