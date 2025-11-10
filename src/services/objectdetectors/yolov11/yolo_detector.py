# objectdetectors/yolov11/yolo_detector.py
from typing import Any, Dict, List
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from src.services.objectdetectors.base_loader import BaseModelLoader
from src.utils.logger import get_logger
logger = get_logger(__name__)

class YoloLoader(BaseModelLoader):
    def __init__(self,  imgsz: int, conf_thres: float, max_batch: int):
        self.weights_path = "src/services/objectdetectors/yolov11/models/yolo11n_custom.pt"
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.max_batch = int(max_batch)
        self.model = None
        self.names = None

    def load(self) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics.YOLO not available")
        self.model = YOLO(self.weights_path)
        self.names = getattr(self.model, "names", None)

        # warmup (best-effort)
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            _ = self.model(dummy)
        except Exception:
            pass
        logger.info(f"[Loader:YOLOv11] loaded {self.weights_path}, imgsz={self.imgsz}")

    def process_frames(self, frames: List[Any]) -> List[Any]:
        return self.model.predict(frames, imgsz=self.imgsz, verbose=False)

    def parse_result(self, res: Any) -> List[Dict]:
        detections: List[Dict] = []
        try:
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                try:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    conf = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    xyxy = getattr(res.boxes, "xyxy", [])
                    conf = getattr(res.boxes, "conf", [])
                    cls = getattr(res.boxes, "cls", [])

                for b, cf, cg in zip(xyxy, conf, cls):
                    if float(cf) < self.conf_thres:
                        continue
                    class_name = (
                        self.names[int(cg)] if self.names and int(cg) in self.names else str(int(cg))
                    )
                    detections.append({
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "class_id": int(cg),
                        "class_name": class_name,
                        "confidence": float(cf)
                    })
        except Exception:
            logger.exception("[Loader:YOLOv11] parse_result failed for one frame")
        return detections