import subprocess
import threading
import time
import cv2
import numpy as np
import queue
import logging
import os
import signal
from typing import Optional, Tuple
from utils.logger import get_logger
log = get_logger(__name__)


class FFmpegStreamCapture:
    """
    A production-grade resilient FFmpeg-based video capturer.
    - Supports HTTPS (HLS) and RTSP with automatic option handling.
    - Auto-reconnects on failure or inactivity.
    - Compatible with motion-based frame skipping (non-blocking reads).
    - Parallel-safe (each stream runs independently).
    """

    def __init__(
            self,
            primary_source: str,
            secondary_source: Optional[str] = None,
            width: int = 640,
            height: int = 480,
            reconnect_interval: float = 2.0,
            watchdog_timeout: float = 8.0,
            queue_size: int = 5,
    ):
        """
        Drop-in compatible with ImprovedStreamCapture:
        - If both URLs are given (RTSP + HTTPS), HTTPS is tried first, RTSP acts as fallback.
        - Handles all width/height and reconnection logic internally.
        """
        # Assign primary and fallback intelligently
        if primary_source and secondary_source:
            if primary_source.startswith("https"):
                self.source = primary_source
                self.fallback_source = secondary_source
            else:
                self.source = secondary_source
                self.fallback_source = primary_source
        else:
            self.source = primary_source or secondary_source
            self.fallback_source = None

        self.width = width
        self.height = height
        self.reconnect_interval = reconnect_interval
        self.watchdog_timeout = watchdog_timeout

        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = threading.Event()
        self.last_frame_time = 0.0
        self.restart_lock = threading.Lock()
        self.retry_attempt = 0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def start(self):
        """Starts FFmpeg process and reading thread."""
        if self.thread and self.thread.is_alive():
            log.info(f"[{self.source}] already running")
            return

        self.running.set()
        self.thread = threading.Thread(target=self._run_capture, daemon=True)
        self.thread.start()
        log.info(f"[FFmpegStreamCapture] started thread for {self.source}")

    def stop(self):
        """Stops capture and cleans up."""
        self.running.clear()
        if self.process:
            self._terminate_ffmpeg()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        log.info(f"[FFmpegStreamCapture] stopped {self.source}")

    def read(self) -> Optional[np.ndarray]:
        """Non-blocking frame read (returns None if no frame available)."""
        try:
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """Checks if process is alive and frames are coming in."""
        if not self.process:
            return False
        alive = (
            self.process.poll() is None
            and (time.time() - self.last_frame_time) < self.watchdog_timeout
        )
        return alive

    # ------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------
    def _run_capture(self):
        """Capture loop with auto-reconnect and watchdog."""
        while self.running.is_set():
            if not self._open_ffmpeg_stream():
                self._handle_reconnect()
                continue

            try:
                frame_size = self.width * self.height * 3
                while self.running.is_set():
                    raw = self.process.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        raise RuntimeError("Incomplete frame read")

                    frame = np.frombuffer(raw, np.uint8).reshape(
                        (self.height, self.width, 3)
                    )

                    # Skip adding if motion-capture logic not triggered externally
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)

                    self.last_frame_time = time.time()

                    # Watchdog timeout check
                    if time.time() - self.last_frame_time > self.watchdog_timeout:
                        log.warning(
                            f"[FFmpegStreamCapture] watchdog: stream not alive, restarting..."
                        )
                        raise RuntimeError("Watchdog timeout")

            except Exception as e:
                log.warning(f"[FFmpegStreamCapture] stream read error: {e}")
            finally:
                self._terminate_ffmpeg()
                self._handle_reconnect()

    def _open_ffmpeg_stream(self) -> bool:
        """Launches the FFmpeg process."""
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
        ]

        # Conditional RTSP options
        if self.source.startswith("rtsp://"):
            ffmpeg_cmd += ["-rtsp_transport", "tcp"]

        # Add resilient network handling
        ffmpeg_cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "2",
            "-timeout", "5000000",
            "-i", self.source,
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-vf", f"scale={self.width}:{self.height}",
            "-",
        ]

        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
                preexec_fn=os.setsid,  # Important for safe termination
            )
            self.last_frame_time = time.time()
            self.retry_attempt = 0
            log.info(f"[FFmpegStreamCapture] opened stream: {self.source}")
            return True
        except Exception as e:
            log.warning(f"[FFmpegStreamCapture] failed to open stream {self.source}: {e}")
            return False

    def _handle_reconnect(self):
        """Handles backoff and fallback logic on failure."""
        with self.restart_lock:
            self.retry_attempt += 1
            delay = self.reconnect_interval * min(self.retry_attempt, 5)
            log.info(
                f"[FFmpegStreamCapture] reconnecting after {delay:.2f}s (attempt={self.retry_attempt})"
            )

            time.sleep(delay)

            # Try fallback if applicable
            if self.fallback_source and self.retry_attempt >= 3:
                log.warning(
                    f"[FFmpegStreamCapture] switching to fallback: {self.fallback_source}"
                )
                self.source, self.fallback_source = self.fallback_source, self.source
                self.retry_attempt = 0

    def _terminate_ffmpeg(self):
        """Gracefully kills FFmpeg process."""
        if not self.process:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception:
            pass
        self.process = None
        log.info(f"[FFmpegStreamCapture] process terminated for {self.source}")
