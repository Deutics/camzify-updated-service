from dataclasses import dataclass
from typing import List, Dict, Any
from src.data_models.frame_data import FrameData


@dataclass
class DetectionResult:
    frame_data: FrameData
    detections: List[Dict[str, Any]]
    processing_time: float