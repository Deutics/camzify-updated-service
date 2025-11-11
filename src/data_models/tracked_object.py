from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    bbox: List[float]          # [x1, y1, x2, y2]
    confidence: float
    stream_id: str
    timestamp: float
    frame_number: int
    history: List[Tuple[float, float]] = None  # list of (x,y) centers over time

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)