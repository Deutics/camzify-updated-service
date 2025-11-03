from dataclasses import dataclass
from typing import List, Dict, Any
from data_models.FrameData import FrameData


@dataclass
class DetectionResult:
    frame_data: FrameData
    detections: List[Dict[str, Any]]
    processing_time: float