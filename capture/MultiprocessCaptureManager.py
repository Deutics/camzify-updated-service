# capture/MultiprocessCaptureManager.py
from typing import Dict, List, Any, Optional, Union
import multiprocessing as mp
import threading
from pathlib import Path

from capture.CaptureWorker import capture_worker
from utils.service_cleanup import safe_join_process, safe_join_thread
from utils.logger import get_logger
logger = get_logger(__name__)

class MultiprocessCaptureManager:
    """Manages multiple capture workers in separate processes or threads."""

    def __init__(
            self,
            streams_config: List[Dict[str, Any]],
            input_queues: Dict[str, mp.Queue],
            target_fps: float = 15.0,
            default_width: int = 640,
            default_height: int = 480,
            use_thread_for_local: bool = True,
    ):

        self.stream_configs = streams_config
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
        sid = cfg["stream_id"]
        rtsp_url = cfg.get("rtsp_url")
        https_url = cfg.get("https_url")
        q = self.queues[sid]
        width = int(cfg.get("camera_width", self.default_width))
        height = int(cfg.get("camera_height", self.default_height))
        per_stream_cfg = self._build_per_stream_cfg(cfg)

        is_thread_mode = (self._is_local_file(rtsp_url) or self._is_local_file(https_url)) and self.use_thread_for_local

        if is_thread_mode:
            th_stop = threading.Event()
            self.thread_stop_events[sid] = th_stop
            t = threading.Thread(
                target=capture_worker,
                args=(rtsp_url, https_url, sid, self.target_fps, q, width, height, per_stream_cfg, th_stop),
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
                args=(rtsp_url, https_url, sid, self.target_fps, q, width, height, per_stream_cfg, p_stop),
                daemon=False,
            )
            p.start()
            self.procs[sid] = p
            logger.info(f"[CAPMAN] Started process for {sid} ({width}x{height})")

    # ---------------- public API ---------------- #

    def start(self):
        with self._lock:
            for cfg in self.stream_configs:
                sid = cfg["stream_id"]
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
        sid = stream_cfg["stream_id"]
        with self._lock:
            if sid in self.procs or sid in self.threads:
                logger.info(f"[CAPMAN] start_stream: {sid} already running")
                return True
            if sid not in self.queues:
                logger.warning(f"[CAPMAN] start_stream: {sid} queue missing")
                return False
            try:
                self._spawn_worker(stream_cfg)
                if not any(s["stream_id"] == sid for s in self.stream_configs):
                    self.stream_configs.append(stream_cfg)
                return True
            except Exception as e:
                logger.exception(f"[CAPMAN] start_stream: failed for {sid}: {e}")
                return False

    def stop_stream(self, stream_id: str) -> bool:
        sid = str(stream_id)
        with self._lock:
            if sid in self.threads:
                ev = self.thread_stop_events.get(sid)
                if ev:
                    ev.set()
                t = self.threads.pop(sid, None)
                self.thread_stop_events.pop(sid, None)
            else:
                t = None

            if sid in self.procs:
                evp = self.proc_stop_events.get(sid)
                if evp:
                    evp.set()
                p = self.procs.pop(sid, None)
                self.proc_stop_events.pop(sid, None)
            else:
                p = None

        if t:
            safe_join_thread(t, timeout=2)
        if p:
            safe_join_process(p, timeout=2)

        with self._lock:
            self.stream_configs = [c for c in self.stream_configs if str(c.get("stream_id")) != sid]

        logger.info(f"[CAPMAN] stop_stream: {sid} stopped")
        return True

    def restart_stream(self, stream_id: str, new_stream_cfg: Optional[Dict[str, Any]] = None) -> bool:
        sid = str(stream_id)
        self.stop_stream(sid)

        cfg_to_use = None
        if new_stream_cfg is not None:
            cfg_to_use = dict(new_stream_cfg)
        else:
            with self._lock:
                for c in self.stream_configs:
                    if str(c.get("stream_id")) == sid:
                        cfg_to_use = dict(c)
                        break

        if cfg_to_use is None:
            logger.warning(f"[CAPMAN] restart_stream: {sid} missing config")
            return False

        ok = self.start_stream(cfg_to_use)
        if ok:
            logger.info(f"[CAPMAN] restart_stream: {sid} restarted")
        else:
            logger.warning(f"[CAPMAN] restart_stream: {sid} failed to start")
        return ok
