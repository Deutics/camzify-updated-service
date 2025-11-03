# SharedInferenceEngine.py
import threading
from typing import List, Tuple, Dict, Any
import numpy as np
import time
from utils.logger import get_logger
logger = get_logger(__name__)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # Keep import failure graceful for environments without ultralytics


class SharedInferenceEngine:
    """
    Shared model instance that accepts mixed batches of frames from multiple streams.

    Usage:
      - start() loads the model
      - process_mixed_batch(frames_with_sid) accepts list of (stream_id, FrameData)
        and returns a mapping { stream_id: [(FrameData, detections), ...] }

    Detections: list of dicts with keys: bbox, class_id, class_name, confidence
    """

    def __init__(self, model_path: str = 'yolov8n.pt', inference_imgsz: int = 320,
                 confidence_threshold: float = 0.25, max_batch_size: int = 4):
        self.model_path = model_path
        self.model = None
        self.lock = threading.Lock()
        self.inference_imgsz = int(inference_imgsz)
        self.confidence_threshold = float(confidence_threshold)
        self.max_batch_size = int(max_batch_size)

    def start(self) -> bool:
        """Load the YOLO model and attempt a best-effort warmup."""
        if YOLO is None:
            logger.info("[SharedInference] ultralytics.YOLO not available (import failed).")
            return False

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.exception(f"[SharedInference] Failed to construct YOLO model: {e}")
            self.model = None
            return False

        # Warmup with a dummy image but don't fail startup if warmup throws
        dummy = np.zeros((self.inference_imgsz, self.inference_imgsz, 3), dtype=np.uint8)
        try:
            with self.lock:
                # Some UL versions accept an ndarray directly
                try:
                    _ = self.model(dummy)
                except Exception:
                    # Best-effort; ignore errors in warmup
                    pass
        except Exception:
            pass

        logger.info(f"[SharedInference] Loaded model {self.model_path}, imgsz={self.inference_imgsz}")
        return True

    def stop(self) -> None:
        """Release model reference (GC will free resources)."""
        self.model = None

    def _process_batch_internal(self, batch: List[Tuple[str, Any]]) -> Dict[str, List[Tuple[Any, List[Dict]]]]:
        """
        Process a single chunk (<= max_batch_size) of frames.
        batch: list of (stream_id, FrameData)
        Returns: mapping { sid: [(FrameData, detections), ...] }
        """
        if not batch:
            return {}

        # Extract frames and keep ordering
        frames = [fd.frame for sid, fd in batch]

        with self.lock:
            try:
                results = self.model.predict(frames, imgsz=self.inference_imgsz, verbose=False)
            except Exception as e:
                logger.exception(f"[SharedInference] inference error: {e}")
                return {}

        outputs: Dict[str, List[Tuple[Any, List[Dict]]]] = {}

        # results should be iterable with same length as frames
        for (sid, fd), res in zip(batch, results):
            detections: List[Dict] = []

            # UL Result object often exposes .boxes with properties xyxy, conf, cls
            try:
                if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                    # Try extracting via tensor -> numpy (works for GPU/CPU)
                    try:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        conf = res.boxes.conf.cpu().numpy()
                        cls = res.boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        # Fallback for older UL versions that provide numpy arrays already
                        try:
                            xyxy = res.boxes.xyxy.numpy()
                            conf = res.boxes.conf.numpy()
                            cls = res.boxes.cls.numpy().astype(int)
                        except Exception:
                            # If extraction fails, skip this result
                            xyxy, conf, cls = [], [], []

                    for b, cf, cg in zip(xyxy, conf, cls):
                        if float(cf) < self.confidence_threshold:
                            continue
                        detections.append({
                            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "class_id": int(cg),
                            "class_name": (self.model.names[int(cg)] if hasattr(self.model, 'names') else str(int(cg))),
                            "confidence": float(cf)
                        })
            except Exception:
                # Defensive: don't fail whole batch for one result
                logger.exception("[SharedInference] warning: failed to parse result for one frame")
                continue

            outputs.setdefault(sid, []).append((fd, detections))

        return outputs

    def process_mixed_batch(self, frames_with_sid: List[Tuple[str, Any]]) -> Dict[str, List[Tuple[Any, List[Dict]]]]:
        """
        Accepts list of (stream_id, FrameData). Processes them in chunks of size <= max_batch_size.
        Returns combined mapping { sid -> [(FrameData, detections), ...] }.
        """
        if not frames_with_sid:
            return {}

        frame_data_detections: Dict[str, List[Tuple[Any, List[Dict]]]] = {}

        # Process in chunks so model can accept up to max_batch_size at a time
        for i in range(0, len(frames_with_sid), self.max_batch_size):
            chunk = frames_with_sid[i: i + self.max_batch_size]
            try:
                part = self._process_batch_internal(chunk)
                for sid, items in part.items():
                    frame_data_detections.setdefault(sid, []).extend(items)
            except Exception as e:
                logger.exception(f"[SharedInference] chunk processing error: {e}")
                continue

        return frame_data_detections