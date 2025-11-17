from dataclasses import dataclass
import numpy as np
import time


@dataclass
class FrameData:
    frame: np.ndarray
    timestamp: float
    stream_id: str
    frame_number: int
    # instance_id:str
