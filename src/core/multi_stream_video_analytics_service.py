import multiprocessing as mp
import threading
import time
from typing import List, Dict, Any

from src.streaming.multiprocess_capture_manager import MultiprocessCaptureManager
from src.utils.priority_thread_safe_queue import PriorityThreadSafeQueue
from src.services.objectdetectors.object_detector import ObjectDetector
from src.services.tracking.tracked_object_manager import TrackedObjectManager
from src.core.rule_config_store import RuleConfigStore
from src.utils.safe_counter import SafeCounter

from src.engine.factories import FEATURE_REGISTRY
from src.engine.rule_engine_manager import RuleEngineManager

from src.services.filtering.stream_filters import is_stream_active, within_time_bounds

from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.utils.utils import start_daemon_thread, safe_close_queue, safe_join_threads
from src.utils.stream_utils import (
    create_stream_components,
    start_forwarder_thread
)
from src.utils.service_helpers import fair_collect_batch


class MultiStreamVideoAnalyticsService:
    """
    Orchestrates multi-stream video analytics pipelines.
    Modular version — logic unchanged, structure improved.
    """

    def __init__(
        self,
        streams_config: Dict[str, Any],
        model_dir: str = "yolov11",
        queue_size: int = 30,
        target_fps: float = 15.0,
        max_mixed_batch: int = 16,
        width: int = 640,
        height: int = 480,
    ):
        self.streams_config = streams_config
        self.target_fps = float(max(1.0, target_fps))

        self.rule_config_store = RuleConfigStore(initial_streams=streams_config)

        self.width = int(width)
        self.height = int(height)

        self.stream_cfg_map = {
            str(sid): dict(cfg) for sid, cfg in streams_config.items()
        }

        self.stream_dims = {
            str(sid): (
                int(cfg.get("camera_width", self.width)),
                int(cfg.get("camera_height", self.height)),
            )
            for sid, cfg in streams_config.items()}

        self.input_queues = {
            str(sid): mp.Queue(maxsize=queue_size)
            for sid in streams_config.keys()
        }
        self.frame_queues = {
            str(sid): PriorityThreadSafeQueue(maxsize=queue_size)
            for sid in streams_config.keys()
        }

        self.capture_manager = MultiprocessCaptureManager(
            streams_config=streams_config,
            input_queues=self.input_queues,
            target_fps=self.target_fps,
            default_width=self.width,
            default_height=self.height,
        )

        self.shared_inference = ObjectDetector(
            model_dir=model_dir,
            inference_imgsz=320,
            confidence_threshold=0.25,
            max_batch_size=int(max_mixed_batch),
        )

        self.object_tracker = TrackedObjectManager()

        self.alert_systems, self.visualizers = {}, {}
        for sid, cfg in streams_config.items():
            sid = str(sid)
            alert, viz, dims = create_stream_components(
                sid, cfg, self.width, self.height
            )
            self.alert_systems[sid] = alert
            self.visualizers[sid] = viz
            self.stream_dims[sid] = dims

        self.rule_engine = RuleEngineManager(
            raw_config_store=self.rule_config_store,
            alert_systems=self.alert_systems,
            visualizers=self.visualizers,
            feature_registry=FEATURE_REGISTRY,
            per_stream_queue_size=256,
            feature_worker_threads=2,
            feature_check_timeout_s=1.0,
        )

        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self.max_mixed_batch = int(max_mixed_batch)
        self._rr_rotation_counter = SafeCounter()
        self._forwarder_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

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
                rule_config_store=self.rule_config_store,
                stop_event=self._stop_event,
                threads_list=self._threads,
                logger=logger,
                is_stream_active_fn=is_stream_active,
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

                tracked_objects = self.object_tracker.update_batch(detections_by_sid)
                frames_for_viz = {sid: frame for sid, frame in frames_with_sid}

                self.rule_engine.dispatch(tracked_objects, frames_for_viz)

            except Exception as e:
                logger.exception(f"[FATAL] batch_loop error: {e}")
                time.sleep(0.01)