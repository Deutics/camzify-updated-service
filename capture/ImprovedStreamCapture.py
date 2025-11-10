import cv2
import time
import threading
from capture.utils import open_opencv_stream, toggle_stream_url

from utils.logger import get_logger

logger = get_logger(__name__)


class ImprovedStreamCapture:
    """
    Advanced OpenCV-based RTSP/HTTPS capture with internal watchdog and auto-recovery.
    Scalable for large multi-stream systems (100+ streams).
    """

    def __init__(
        self,
        rtsp_url: str,
        https_url: str = None,
        width: int = 640,
        height: int = 480,
        reconnect_delay: float = 2.0,
        max_restart_attempts: int = 3,
        freeze_timeout: float = 15.0,
        watchdog_interval: float = 3.0,
    ):
        self._restart_lock = threading.Lock()
        self.rtsp_url = rtsp_url
        self.https_url = https_url
        self.current_url = https_url or rtsp_url
        self.using_https = https_url is not None

        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.max_restart_attempts = max_restart_attempts
        self.freeze_timeout = freeze_timeout
        self.watchdog_interval = watchdog_interval

        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.restart_attempt = 0
        self.last_frame_time = 0
        self.running = False

        self._open_stream()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.capture_thread.start()
        self.watchdog_thread.start()

    # ---------------------------------------------------------------------- #
    #                          Internal Methods                              #
    # ---------------------------------------------------------------------- #

    def _open_stream(self):
        """Open a new OpenCV stream connection."""
        with self._restart_lock:
            self.release()
            self.cap = open_opencv_stream(self.current_url, self.width, self.height)
            self.last_frame_time = time.time()
            self.running = True
            logger.info(f"Stream opened: {self.current_url}")

    def _switch_url(self):
        """Toggle between RTSP and HTTPS after repeated failures."""
        old_url = self.current_url
        self.current_url, self.using_https = toggle_stream_url(
            self.rtsp_url, self.https_url, self.using_https
        )
        logger.warning(f"Switched URL from {old_url} to {self.current_url}")

    def _capture_loop(self):
        """Continuously read frames from stream (background thread)."""
        while True:
            if not self.running:
                time.sleep(0.5)
                continue

            if self.cap is None or not self.cap.isOpened():
                logger.warning("Stream not open, reopening...")
                self._open_stream()
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.restart_attempt += 1
                logger.warning(f"Read failed ({self.restart_attempt})")

                if self.restart_attempt >= self.max_restart_attempts:
                    self._switch_url()
                    self.restart_attempt = 0

                time.sleep(self.reconnect_delay)
                self._open_stream()
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            with self.frame_lock:
                self.latest_frame = frame
            self.last_frame_time = time.time()
            self.restart_attempt = 0
            time.sleep(0.01)

    def _watchdog_loop(self):
        """Monitors stream for inactivity and restarts if frozen."""
        while True:
            time.sleep(self.watchdog_interval)
            if not self.running:
                continue

            if time.time() - self.last_frame_time > self.freeze_timeout:
                logger.warning(
                    f"Watchdog: stream frozen for {self.freeze_timeout}s. Restarting..."
                )
                self._open_stream()

    # ---------------------------------------------------------------------- #
    #                           Public API                                   #
    # ---------------------------------------------------------------------- #

    def read(self):
        """Non-blocking: return most recent frame or None if unavailable."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_alive(self) -> bool:
        """Check if the capture is active and recently produced frames."""
        if not self.cap or not self.cap.isOpened():
            return False
        return (time.time() - self.last_frame_time) < self.freeze_timeout

    def release(self):
        """Safely release capture resources."""
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                logger.exception("Error releasing capture")
        self.cap = None
        logger.info("Stream released")
