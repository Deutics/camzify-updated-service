# data_models/stream_config_data.py
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class StreamConfig:
    """
    Normalized stream configuration used by CaptureManager / CaptureWorker.
    Keep fields minimal and explicit. Additional per-stream options can be
    stored in the `meta` dict.
    """
    stream_id: str
    rtsp_url: Optional[str] = None
    https_url: Optional[str] = None
    camera_width: int = 640
    camera_height: int = 480
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}