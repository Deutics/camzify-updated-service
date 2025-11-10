#rule_engine/StreamRuleEngine.py

import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any, List, Optional, Tuple

from .FeatureBase import FeatureBase
from .factories import get_feature_factory
from data_models.FrameData import FrameData
from data_models.DetectionResults import DetectionResult
from data_models.TrackedObject import TrackedObject
from .ConfigStoreAdapter import ConfigStoreAdapter
from utils.logger import get_logger  #  centralized logger
from .utils import safe_call, extract_overlay_geometry  # helpers
logger = get_logger(__name__)


class StreamRuleEngine:
    """
    Per-stream rules engine (modernized):
      - Maintains a bounded queue of (tracked_objects, frame_data)
      - Holds feature instances keyed by (feature_name, index)
      - Uses fine-grained updates via RuleEngineManager
      - Evaluates features in thread pool, pushes geometry to visualizer, alerts to alert_system
    """

    def __init__(
        self,
        stream_id: str,
        config_store: ConfigStoreAdapter,
        feature_registry: Dict[str, Any],
        alert_system: Any = None,
        visualizer: Any = None,
        queue_maxsize: int = 256,
        feature_worker_threads: int = 4,
        feature_check_timeout_s: float = 1.5,
    ):
        self.stream_id = str(stream_id)
        self.config_store = config_store
        self.feature_registry = feature_registry
        self.alert_system = alert_system
        self.visualizer = visualizer

        self.queue: queue.Queue = queue.Queue(maxsize=int(queue_maxsize))
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.executor = ThreadPoolExecutor(max_workers=int(feature_worker_threads))
        self.feature_check_timeout_s = float(feature_check_timeout_s)

        self.feature_instances: Dict[Tuple[str, int], FeatureBase] = {}
        self._features_lock = threading.Lock()

        safe_call(self._load_features_from_config)
        self.worker_thread.start()

    # -----------------------------
    # Public API
    # -----------------------------
    def enqueue(self, tracked_objects: List[TrackedObject], frame_data: Optional[FrameData] = None, meta: Optional[dict] = None) -> bool:
        item = {"tracked_objects": tracked_objects, "frame_data": frame_data, "meta": meta or {}}
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
                self.queue.put_nowait(item)
                return True
            except Exception as e:
                logger.warning(f"Queue full, failed to enqueue: {e}")
                return False

    def refresh_rules(self) -> None:
        """Force a refresh of rules from the config store."""
        safe_call(self._on_config_update)

    def stop(self) -> None:
        self.stop_event.set()
        safe_call(lambda: self.queue.put_nowait({"tracked_objects": [], "meta": {"_stop": True}}))
        safe_call(lambda: self.worker_thread.join(timeout=1.0))

        with self._features_lock:
            for inst in self.feature_instances.values():
                safe_call(inst.shutdown)
            self.feature_instances.clear()

        safe_call(lambda: self.executor.shutdown(wait=False))

    # -----------------------------
    # Internal: config handling
    # -----------------------------
    def _load_features_from_config(self) -> None:
        cfgs = safe_call(lambda: self.config_store.get_active_features(self.stream_id) or {}, default={})
        new_instances: Dict[Tuple[str, int], FeatureBase] = {}

        for feature_name, cfg_list in cfgs.items():
            factory = self.feature_registry.get(feature_name) or get_feature_factory(feature_name)
            if not factory:
                continue
            for idx, cfg in enumerate(cfg_list):
                key = (feature_name, idx)
                inst = safe_call(lambda: factory(cfg, self.stream_id, self.alert_system, self.visualizer))
                if inst:
                    new_instances[key] = inst

        with self._features_lock:
            # Shutdown removed instances
            for k in set(self.feature_instances.keys()) - set(new_instances.keys()):
                safe_call(lambda inst=self.feature_instances[k]: inst.shutdown())
            self.feature_instances = new_instances

    def _on_config_update(self) -> None:
        logger.info("Applying rules update")
        current_cfgs = safe_call(lambda: self.config_store.get_active_features(self.stream_id) or {}, default={})

        with self._features_lock:
            new_instances: Dict[Tuple[str, int], FeatureBase] = {}

            for feature_name, cfg_list in current_cfgs.items():
                factory = self.feature_registry.get(feature_name) or get_feature_factory(feature_name)
                if not factory:
                    continue

                for idx, cfg in enumerate(cfg_list):
                    key = (feature_name, idx)
                    existing = self.feature_instances.get(key)

                    if existing and hasattr(existing, "update_config"):
                        if safe_call(lambda: existing.update_config(cfg)):
                            new_instances[key] = existing
                            continue

                    # Recreate if needed
                    if existing:
                        safe_call(existing.shutdown)
                    inst = safe_call(lambda: factory(cfg, self.stream_id, self.alert_system, self.visualizer))
                    if inst:
                        new_instances[key] = inst

            # Shutdown removed instances
            for old_key, old_inst in self.feature_instances.items():
                if old_key not in new_instances:
                    safe_call(old_inst.shutdown)

            self.feature_instances = new_instances

            # Update visualizer overlay
            overlay = extract_overlay_geometry(list(self.feature_instances.values()))
            if self.visualizer and hasattr(self.visualizer, "set_overlay_geometry"):
                safe_call(lambda: self.visualizer.set_overlay_geometry(self.stream_id, overlay))
                logger.debug(f"Overlay updated: {overlay}")

    # -----------------------------
    # Internal: worker loop
    # -----------------------------
    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                time.sleep(0.01)
                continue

            tracked_objects: List[TrackedObject] = item.get("tracked_objects") or []
            frame_data: Optional[FrameData] = item.get("frame_data")
            meta = item.get("meta", {})

            with self._features_lock:
                feature_items = list(self.feature_instances.items())

            for feature_key, feature_inst in feature_items:
                def _process_feature():
                    events = None
                    try:
                        future_ = self.executor.submit(feature_inst.check_intrusion, tracked_objects, frame_data, meta)
                        try:
                            events = future_.result(timeout=self.feature_check_timeout_s)
                        except TimeoutError:
                            future_.cancel()
                            logger.warning(f"Feature {feature_key} timed out")
                    except Exception as e:
                        logger.exception(f"Feature {feature_key} check_intrusion failed: {e}")
                    return events

                events = _process_feature()

                # Geometry for visualization
                geom_list = extract_overlay_geometry([feature_inst])
                if self.visualizer and hasattr(self.visualizer, "update_frame_visualization"):
                    safe_call(lambda: self.visualizer.update_frame_visualization(
                        frame_data.frame if frame_data else None,
                        tracked_objects,
                        events or [],
                        geom_list,
                    ))

                # Alerts only when events exist
                if events and self.alert_system and hasattr(self.alert_system, "handle_intrusion"):
                    safe_call(lambda: self._send_alert(tracked_objects, frame_data, events, feature_inst))

        safe_call(lambda: self.executor.shutdown(wait=False))

    def _send_alert(self, tracked_objects, frame_data, events, feature_inst):
        dets = [{
            "bbox": obj.bbox,
            "class_id": obj.class_id,
            "class_name": obj.class_name,
            "confidence": obj.confidence,
            "track_id": obj.track_id,
        } for obj in tracked_objects]
        detection_results = DetectionResult(frame_data=frame_data, detections=dets, processing_time=0.0)
        geometry = getattr(feature_inst, "line_points", None) or getattr(feature_inst, "zone_points", None)
        self.alert_system.handle_intrusion(detection_results, events, geometry)
