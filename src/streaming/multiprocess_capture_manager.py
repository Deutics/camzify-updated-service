# src/streaming/MultiprocessCaptureManager.py
from typing import Dict, Any, Optional
import multiprocessing as mp
import threading
from pathlib import Path

from src.streaming.capture_worker import capture_worker
from src.utils.service_helpers import safe_join_process, safe_join_thread
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MultiprocessCaptureManager:
    """Manages multiple capture workers in separate processes or threads."""

    def __init__(
        self,
        streams_config: Dict[str, Any],
        input_queues: Dict[str, mp.Queue],
        target_fps: float = 15.0,
        default_width: int = 640,
        default_height: int = 480,
        use_thread_for_local: bool = True,
    ):

        self.stream_configs = streams_config  # keyed by stream_id
        self.queues = input_queues
        self.target_fps = float(target_fps)
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.use_thread_for_local = bool(use_thread_for_local)

        self.procs: Dict[str, mp.Process] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.stop_event = mp.Event()
        self.thread_stop_events: Dict[str, threading.Event] = {}
        self.proc_stop_events: Dict[str, mp.Event] = {}
        self._lock = threading.Lock()

    # ------------------ helpers ------------------ #

    @staticmethod
    def _is_local_file(url: Optional[str]) -> bool:
        if not isinstance(url, str):
            return False
        if url.startswith(("rtsp://", "http://", "https://")):
            return False
        return Path(url).expanduser().is_file()

    @staticmethod
    def _build_per_stream_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sensor_fps": cfg.get("sensor_fps", 2.0),
            "pre_buffer_len": cfg.get("pre_buffer_len", 3),
            "motion_inactive_timeout": cfg.get("motion_inactive_timeout", 1.5),
            "max_active_seconds": cfg.get("max_active_seconds", 10.0),
            "motion_min_area": cfg.get("motion_min_area", 300),
            "motion_history_frames": cfg.get("motion_history_frames", 20),
            "motion_threshold": cfg.get("motion_threshold", 20),
            "jpeg_quality": cfg.get("jpeg_quality", 60),
            "motion_w": cfg.get("motion_w", 160),
            "motion_h": cfg.get("motion_h", 90),
        }

    def _spawn_worker(self, cfg: Dict[str, Any]):
        sid = str(cfg["stream_id"])
        rtsp_url = cfg.get("rtsp_url")
        https_url = cfg.get("https_url")
        input_q = self.queues[sid]
        width = int(cfg.get("camera_width", self.default_width))
        height = int(cfg.get("camera_height", self.default_height))
        per_stream_cfg = self._build_per_stream_cfg(cfg)

        is_thread_mode = (self._is_local_file(rtsp_url) or self._is_local_file(https_url)) and self.use_thread_for_local

        if is_thread_mode:
            th_stop = threading.Event()
            self.thread_stop_events[sid] = th_stop
            t = threading.Thread(
                target=capture_worker,
                args=(rtsp_url, https_url, sid, self.target_fps, input_q, width, height, per_stream_cfg, th_stop),
                daemon=True,
            )
            t.start()
            self.threads[sid] = t
            logger.info(f"[CAPMAN] Started thread for {sid} ({width}x{height})")
        else:
            p_stop = mp.Event()
            self.proc_stop_events[sid] = p_stop
            p = mp.Process(
                target=capture_worker,
                args=(rtsp_url, https_url, sid, self.target_fps, input_q, width, height, per_stream_cfg, p_stop),
                daemon=False,
            )
            p.start()
            self.procs[sid] = p
            logger.info(f"[CAPMAN] Started process for {sid} ({width}x{height})")

    # ---------------- public API ---------------- #

    def start(self):
        with self._lock:
            for cfg in self.stream_configs.values():
                sid = str(cfg["stream_id"])
                if sid in self.procs or sid in self.threads:
                    continue
                self._spawn_worker(cfg)

    def stop(self):
        with self._lock:
            self.stop_event.set()
            for ev in self.thread_stop_events.values():
                ev.set()
            for ev in self.proc_stop_events.values():
                ev.set()

        for q in self.queues.values():
            try:
                q.put_nowait(None)
            except Exception:
                pass

        for p in list(self.procs.values()):
            safe_join_process(p, timeout=2)
        for t in list(self.threads.values()):
            safe_join_thread(t, timeout=2)

        with self._lock:
            self.procs.clear()
            self.threads.clear()
            self.thread_stop_events.clear()
            self.proc_stop_events.clear()

        logger.info("[CAPMAN] All capture workers stopped.")

    def start_stream(self, stream_cfg: Dict[str, Any]) -> bool:
        sid = str(stream_cfg["stream_id"])
        with self._lock:
            if sid in self.procs or sid in self.threads:
                logger.info(f"[CAPMAN] start_stream: {sid} already running")
                return True
            if sid not in self.queues:
                logger.warning(f"[CAPMAN] start_stream: {sid} queue missing")
                return False
            try:
                self._spawn_worker(stream_cfg)
                self.stream_configs[sid] = dict(stream_cfg)
                return True
            except Exception as e:
                logger.exception(f"[CAPMAN] start_stream: failed for {sid}: {e}")
                return False