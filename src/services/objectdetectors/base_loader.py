# objectdetectors/base_loader.py
from typing import Any, Dict, List, Tuple

class BaseModelLoader:
    """
    All concrete loaders must implement:
      - __init__(weights_path: str, imgsz: int, conf_thres: float, max_batch: int)
      - load() -> None   # build the model and warmup if needed
      - infer(frames: List[Any]) -> List[Any]  # backend-specific result list aligned with frames
      - parse_result(back_result: Any) -> List[Dict]  # return list of detections {bbox, class_id, class_name, confidence}
      - names (optional attribute) -> mapping id->name
    """
    def load(self) -> None:
        raise NotImplementedError

    def process_frames(self, frames: List[Any]) -> List[Any]:
        raise NotImplementedError

    def parse_result(self, back_result: Any) -> List[Dict]:
        raise NotImplementedError