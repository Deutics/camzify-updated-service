import multiprocessing as mp
import threading
import time
import os
from typing import List, Dict, Any

# Core components
from capture.MultiprocessCaptureManager import MultiprocessCaptureManager
from PriorityThreadSafeQueue import PriorityThreadSafeQueue
from SharedInferenceEngine import SharedInferenceEngine
from tracking.TrackedObjectManager import TrackedObjectManager
from RuleConfigStore import RuleConfigStore
from data_models.FrameData import FrameData
from SafeCounter import SafeCounter

# Rule engine
from rule_engine.factories import FEATURE_REGISTRY
from rule_engine.RuleEngineManager import RuleEngineManager

# Filters
from filters.SizeThresholdCache import SizeThresholdCache
from filters.StreamFilters import is_stream_active, within_time_bounds

# Centralized logger
from utils.logger import get_logger
logger = get_logger(__name__)

# Utility imports
from utils.utils import start_daemon_thread, safe_close_queue, safe_join_threads
from utils.stream_utils import (
    create_stream_components,
    start_forwarder_thread,
    stop_stream_resources,
)
from utils.service_helpers import (
    fair_collect_batch,
    parallel_filter_detections,
)


class MultiStreamVideoAnalyticsService:
    """
    Orchestrates multi-stream video analytics pipelines.
    Modular version — logic unchanged, structure improved.
    """

    def __init__(
        self,
        streams_config: List[Dict[str, Any]],
        rules_config: Dict[str, Any],
        model_path: str = "yolov8n.pt",
        queue_size: int = 30,
        target_fps: float = 15.0,
        max_mixed_batch: int = 16,
        width: int = 640,
        height: int = 480,
    ):
        self.streams_config = streams_config
        self.target_fps = float(max(1.0, target_fps))

        # Unified rule/stream store
        self.rule_config_store = RuleConfigStore(
            initial_rules=rules_config, initial_streams=streams_config
        )

        # Global frame defaults
        self.width = int(width)
        self.height = int(height)

        # Stream-level config maps
        self.stream_cfg_map = {
            str(cfg["stream_id"]): dict(cfg) for cfg in streams_config
        }
        self.stream_dims = {
            str(cfg["stream_id"]): (
                int(cfg.get("camera_width", self.width)),
                int(cfg.get("camera_height", self.height)),
            )
            for cfg in streams_config
        }

        # Queues
        self.input_queues = {
            str(cfg["stream_id"]): mp.Queue(maxsize=queue_size)
            for cfg in streams_config
        }
        self.frame_queues = {
            str(cfg["stream_id"]): PriorityThreadSafeQueue(maxsize=queue_size)
            for cfg in streams_config
        }

        # Core components
        self.capture_manager = MultiprocessCaptureManager(
            streams_config=streams_config,
            input_queues=self.input_queues,
            target_fps=self.target_fps,
            default_width=self.width,
            default_height=self.height,
        )

        self.shared_inference = SharedInferenceEngine(
            model_path=model_path,
            inference_imgsz=320,
            confidence_threshold=0.25,
            max_batch_size=int(max_mixed_batch),
        )

        self.object_tracker = TrackedObjectManager()

        # Per-stream alert + visualization
        self.alert_systems, self.visualizers = {}, {}
        for cfg in streams_config:
            sid = str(cfg["stream_id"])
            alert, viz, dims = create_stream_components(
                sid, cfg, self.width, self.height
            )
            self.alert_systems[sid] = alert
            self.visualizers[sid] = viz
            self.stream_dims[sid] = dims

        # Rule engine
        self.rule_engine = RuleEngineManager(
            raw_config_store=self.rule_config_store,
            alert_systems=self.alert_systems,
            visualizers=self.visualizers,
            feature_registry=FEATURE_REGISTRY,
            per_stream_queue_size=256,
            feature_worker_threads=2,
            feature_check_timeout_s=1.0,
        )

        # Threshold cache
        self.threshold_cache = SizeThresholdCache(
            self.rule_config_store,
            get_stream_dims_fn=lambda sid: self.stream_dims.get(
                str(sid), (self.width, self.height)
            ),
            default_frame_size=(self.width, self.height),
        )
        self.threshold_cache.attach_to_streams(list(self.stream_cfg_map.keys()))

        # Internal state
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self.max_mixed_batch = int(max_mixed_batch)
        self._rr_rotation_counter = SafeCounter()
        self._forwarder_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        # Subscriptions
        self._register_config_callbacks()

    # ---------------------------------------------------------
    # Callback Subscriptions
    # ---------------------------------------------------------
    def _register_config_callbacks(self):
        self.rule_config_store.on_stream_added(self._on_stream_added)
        self.rule_config_store.on_stream_removed(self._on_stream_removed)
        self.rule_config_store.on_stream_updated(self._on_stream_updated)
        self.rule_config_store.on_rules_updated(self._on_rules_updated)
        logger.info("Subscribed to config store events")

    # ---------------------------------------------------------
    # Stream Lifecycle Handlers
    # ---------------------------------------------------------
    def _on_stream_added(self, stream_id: str, new_stream_cfg: Dict[str, Any]):
        sid = str(stream_id)
        try:
            with self._lock:
                if sid in self.stream_cfg_map:
                    logger.debug(f"Stream {sid} already exists — ignoring.")
                    return

                self.stream_cfg_map[sid] = dict(new_stream_cfg)
                alert, viz, dims = create_stream_components(
                    sid, new_stream_cfg, self.width, self.height
                )
                self.alert_systems[sid] = alert
                self.visualizers[sid] = viz
                self.stream_dims[sid] = dims

                # Queues
                self.input_queues[sid] = mp.Queue(maxsize=30)
                self.frame_queues[sid] = PriorityThreadSafeQueue(maxsize=30)
                self._init_roundrobin_order()

            # Capture worker
            ok = self.capture_manager.start_stream(new_stream_cfg)
            if not ok:
                logger.warning(f"Failed to start capture for stream {sid}")
                return

            # Forwarder thread
            self._forwarder_threads[sid] = start_forwarder_thread(
                sid=sid,
                in_queue=self.input_queues[sid],
                out_queue=self.frame_queues[sid],
                stream_cfg=self.stream_cfg_map[sid],
                store=self.rule_config_store,
                stop_event=self._stop_event,
                threads_list=self._threads,
                logger=logger,
                is_stream_active_fn=is_stream_active,
                within_time_bounds_fn=within_time_bounds,
            )

            # Attach threshold
            self.threshold_cache.attach_to_streams([sid])

            # Start visualizer
            start_daemon_thread(
                self.visualizers[sid].start_display,
                name=f"viz-{sid}",
                threads_list=self._threads,
            )

            logger.info(f"Stream {sid} added successfully")

        except Exception as e:
            logger.error(f"on_stream_added error: {e}", exc_info=True)

    def _on_stream_removed(self, stream_id: str):
        sid = str(stream_id)
        try:
            with self._lock:
                frame_q = self.frame_queues.get(sid)
                viz = self.visualizers.pop(sid, None)
                self._forwarder_threads.pop(sid, None)

                stop_stream_resources(
                    self.capture_manager,
                    viz,
                    self.threshold_cache,
                    frame_q,
                    logger,
                )

                for d in [self.stream_cfg_map, self.stream_dims,
                          self.input_queues, self.frame_queues]:
                    d.pop(sid, None)

                self._init_roundrobin_order()

            logger.info(f"Stream {sid} removed successfully")

        except Exception as e:
            logger.error(f"on_stream_removed error: {e}", exc_info=True)

    def _on_stream_updated(
        self, stream_id: str, old_cfg: Dict[str, Any], new_cfg: Dict[str, Any], diff: Dict[str, Any]
    ):
        sid = str(stream_id)
        try:
            with self._lock:
                self.stream_cfg_map[sid] = dict(new_cfg)
                w = int(new_cfg.get("camera_width", self.width))
                h = int(new_cfg.get("camera_height", self.height))
                old_dims = self.stream_dims.get(sid)
                self.stream_dims[sid] = (w, h)

            changed = set(diff.get("changed_keys", []))
            url_changed = any(k in changed for k in ("rtsp_url", "https_url"))
            dims_changed = (old_dims != (w, h))

            if url_changed or dims_changed:
                if not self.capture_manager.restart_stream(sid, new_cfg):
                    logger.warning(f"Restart failed for stream {sid}")
                if dims_changed:
                    self.threshold_cache.invalidate(sid)

            logger.info(
                f"Stream {sid} updated (restart={url_changed or dims_changed}, keys={list(changed)})"
            )

        except Exception as e:
            logger.error(f"on_stream_updated error: {e}", exc_info=True)

    def _on_rules_updated(
        self, stream_id: str, old_rules: Dict[str, Any], new_rules: Dict[str, Any], diff: Dict[str, Any]
    ):
        sid = str(stream_id)
        try:
            changed_keys = set(diff.get("changed_keys", []))
            if changed_keys:
                self.threshold_cache.invalidate(sid)
            logger.info(f"Rules updated for stream {sid}: {list(changed_keys)}")
        except Exception as e:
            logger.error(f"on_rules_updated error: {e}", exc_info=True)

    # ---------------------------------------------------------
    # Lifecycle Control
    # ---------------------------------------------------------
    def start(self) -> bool:
        if not self.shared_inference.start():
            return False

        self.capture_manager.start()

        for sid in list(self.input_queues.keys()):
            self._forwarder_threads[sid] = start_forwarder_thread(
                sid=sid,
                in_queue=self.input_queues[sid],
                out_queue=self.frame_queues[sid],
                stream_cfg=self.stream_cfg_map[sid],
                store=self.rule_config_store,
                stop_event=self._stop_event,
                threads_list=self._threads,
                logger=logger,
                is_stream_active_fn=is_stream_active,
                within_time_bounds_fn=within_time_bounds,
            )

        for sid, viz in self.visualizers.items():
            start_daemon_thread(viz.start_display, name=f"viz-{sid}", threads_list=self._threads)

        start_daemon_thread(self._batch_loop, name="batch-loop", threads_list=self._threads)

        logger.info("Service started successfully.")
        return True

    def stop(self):
        self._stop_event.set()

        for q in self.frame_queues.values():
            safe_close_queue(q)

        try:
            self.capture_manager.stop()
            self.threshold_cache.detach()
            self.shared_inference.stop()
            self.rule_engine.stop()
        except Exception as e:
            logger.debug(f"Error during stop: {e}", exc_info=True)

        for viz in self.visualizers.values():
            try:
                viz.stop()
            except Exception:
                pass

        safe_join_threads(self._threads, timeout=0.5)
        logger.info("Service stopped cleanly.")

    # ---------------------------------------------------------
    # Batch Processing Loop
    # ---------------------------------------------------------
    def _init_roundrobin_order(self):
        self._rr_keys = list(self.frame_queues.keys())

    def _batch_loop(self):
        while not self._stop_event.is_set():
            try:
                per_stream_cap = max(1, min(6, int(self.max_mixed_batch // max(1, len(self.frame_queues)))))
                frames_with_sid, per_stream_counts = fair_collect_batch(
                    frame_queues=self.frame_queues,
                    rr_counter=self._rr_rotation_counter,
                    rr_keys=getattr(self, "_rr_keys", []),
                    per_stream_cap=per_stream_cap,
                    tick_timeout=0.005,
                )

                if not frames_with_sid:
                    time.sleep(0.005)
                    continue

                frames_with_sid.sort(key=lambda x: (x[0], getattr(x[1], "frame_number", 0)))
                logger.debug(f"[BATCH] frames={len(frames_with_sid)} breakdown={per_stream_counts}")

                detections_by_sid = self.shared_inference.process_mixed_batch(frames_with_sid)

                thresholds_by_sid = {
                    sid: self.threshold_cache.get(sid)
                    for sid, _ in frames_with_sid
                }
                filtered_detections = parallel_filter_detections(
                    detections_by_sid, thresholds_by_sid, logger
                )

                tracked_objects = self.object_tracker.update_batch(filtered_detections)
                frames_for_viz = {sid: frame for sid, frame in frames_with_sid}

                self.rule_engine.dispatch(tracked_objects, frames_for_viz)

            except Exception as e:
                logger.exception(f"[FATAL] batch_loop error: {e}")
                time.sleep(0.01)
