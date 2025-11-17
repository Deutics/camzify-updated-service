# rule_engine/RuleEngineManager.py

import threading
from typing import Dict, Any, Optional, List
from src.engine.stream_rule_engine import StreamRuleEngine
from src.utils.logger import get_logger
logger = get_logger(__name__)

class RuleEngineManager:
    """
    Manages per-stream StreamRuleEngine instances.
    Subscribes to RuleConfigStore fine-grained events and triggers targeted refreshes.
    """

    def __init__(
        self,
        raw_config_store: Any,
        alert_systems: Dict[str, Any],
        visualizers: Dict[str, Any],
        feature_registry: Optional[Dict[str, Any]] = None,
        per_stream_queue_size: int = 256,
        feature_worker_threads: int = 4,
        feature_check_timeout_s: float = 1.5,

    ):

        self.config_store = raw_config_store

        self.alert_systems = alert_systems or {}
        self.visualizers = visualizers or {}
        self.feature_registry = feature_registry or {}

        self.per_stream_queue_size = int(per_stream_queue_size)
        self.feature_worker_threads = int(feature_worker_threads)
        self.feature_check_timeout_s = float(feature_check_timeout_s)

        self._lock = threading.Lock()
        self._stream_engines: Dict[str, StreamRuleEngine] = {}



    # ----------------------------------------------------------------------
    def _ensure_stream_engine(self, stream_id: str) -> StreamRuleEngine:
        sid = str(stream_id)
        with self._lock:
            eng = self._stream_engines.get(sid)
            if eng is not None:
                logger.debug(f"[RuleEngineManager] found existing engine for {sid}")
                return eng

            logger.info(f"[RuleEngineManager] creating StreamRuleEngine for {sid}")

            eng = StreamRuleEngine(
                stream_id=sid,
                config_store=self.config_store,
                feature_registry=self.feature_registry,
                alert_system=self.alert_systems.get(sid),
                visualizer=self.visualizers.get(sid),
                queue_maxsize=self.per_stream_queue_size,
                feature_worker_threads=self.feature_worker_threads,
                feature_check_timeout_s=self.feature_check_timeout_s,
            )
            self._stream_engines[sid] = eng
            return eng

    # ----------------------------------------------------------------------
    def dispatch(
        self,
        tracked_objects_by_stream: Dict[str, List[Any]],
        frames_for_viz: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        logger.debug(f"[RuleEngineManager] dispatch called for sids={list(tracked_objects_by_stream.keys())}")
        frames_for_viz = frames_for_viz or {}
        results: Dict[str, bool] = {}

        for sid, tracked_objects in tracked_objects_by_stream.items():
            try:
                num_objs = len(tracked_objects)
                logger.debug(
                    f"[RuleEngineManager] Stream {sid}: {num_objs} tracked objects passed to dispatcher."
                )
                engine = self._ensure_stream_engine(sid)
                frame_data = frames_for_viz.get(sid)
                results[sid] = engine.enqueue(tracked_objects, frame_data=frame_data, meta={})
            except Exception as e:
                logger.exception(f"[RuleEngineManager] Dispatch error for stream {sid}: {e}")
                results[sid] = False

        return results

    def stop(self) -> None:
        with self._lock:
            engines = list(self._stream_engines.values())
            self._stream_engines.clear()
        for eng in engines:
            try:
                eng.stop()
            except Exception:
                logger.exception("Failed stopping one of the engines")
        logger.info("All engines stopped successfully")
