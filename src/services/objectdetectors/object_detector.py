# objectdetectors/object_detector.py
import importlib
import threading
from typing import List, Tuple, Dict, Any
import time
from src.utils.logger import get_logger
from config.constants import Config

logger = get_logger(__name__)

class ObjectDetector:
    """
    Shared inference engine that dynamically loads a backend from MODEL_REGISTRY.

    - start(): loads backend loader and warms up
    - process_mixed_batch(frames_with_sid): delegates batching to backend.infer + parse_result
    """

    def __init__(
        self,
        model_dir: str = "yolov11",
        # weights_path: str | None = None,
        inference_imgsz: int = 320,
        confidence_threshold: float = 0.25,
        max_batch_size: int = 4,
        # loader_module: str | None = None,   # optional manual override
        # loader_class: str | None = None,    # optional manual override
    ):
        self.model_dir = model_dir
        # self.weights_path = weights_path
        self.inference_imgsz = int(inference_imgsz)
        self.confidence_threshold = float(confidence_threshold)
        self.max_batch_size = int(max_batch_size)

        # self._loader_module = loader_module
        # self._loader_class = loader_class

        self.loader = None  # concrete BaseModelLoader instance
        self.lock = threading.Lock()

    def _resolve_loader_spec(self) -> tuple[str, str]:
        """
        Returns (module_path, class_name, weights_path). Uses registry unless overridden manually.
        """
        # if self._loader_module and self._loader_class:
        #     # Manual override; weights_path may still default
        #     weights = self.weights_path or ""
        #     return self._loader_module, self._loader_class, weights

        reg = Config.MODEL_REGISTRY.get(self.model_dir)
        if not reg:
            raise ValueError(f"Unknown model_key '{self.model_dir}' not found in MODEL_REGISTRY")

        module = reg["module"]
        class_name = reg["class"]

        return module, class_name

    def start(self) -> bool:
        """Dynamically import loader, construct, and load model."""
        try:
            module_path, class_name = self._resolve_loader_spec()
            mod = importlib.import_module(module_path)
            LoaderCls = getattr(mod, class_name)
            self.loader = LoaderCls(
                imgsz=self.inference_imgsz,
                conf_thres=self.confidence_threshold,
                max_batch=self.max_batch_size,
            )
            self.loader.load()
            logger.info(f"[SharedInference] Loaded backend {self.model_dir} via {module_path}.{class_name}")
            return True
        except Exception as e:
            logger.exception(f"[SharedInference] Failed to load backend '{self.model_dir}': {e}")
            self.loader = None
            return False

    def stop(self) -> None:
        self.loader = None  # GC will handle; customize if backend needs explicit release

    def _process_batch_internal(self, batch: List[Tuple[str, Any]]) -> Dict[str, List[Tuple[Any, List[Dict]]]]:
        if not batch:
            return {}

        frames = [fd.frame for _, fd in batch]

        with self.lock:
            try:
                results = self.loader.process_frames(frames)  # backend-specific result objects
            except Exception as e:
                logger.exception(f"[SharedInference] inference error: {e}")
                return {}

        outputs: Dict[str, List[Tuple[Any, List[Dict]]]] = {}
        for (sid, fd), res in zip(batch, results):
            try:
                detections = self.loader.parse_result(res)  # normalized detections list
                outputs.setdefault(sid, []).append((fd, detections))
            except Exception:
                logger.exception("[SharedInference] parse_result failed for one frame")
                continue

        return outputs

    def process_mixed_batch(self, frames_with_sid: List[Tuple[str, Any]]) -> Dict[str, List[Tuple[Any, List[Dict]]]]:
        if not frames_with_sid:
            return {}

        detection_with_sid: Dict[str, List[Tuple[Any, List[Dict]]]] = {}
        for i in range(0, len(frames_with_sid), self.max_batch_size):
            chunk = frames_with_sid[i: i + self.max_batch_size]
            try:
                part = self._process_batch_internal(chunk)
                for sid, items in part.items():
                    detection_with_sid.setdefault(sid, []).extend(items)
            except Exception as e:
                logger.exception(f"[SharedInference] chunk processing error: {e}")
                continue
        return detection_with_sid