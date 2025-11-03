import time
from collections import deque
import cv2
import numpy as np
import multiprocessing as mp
import threading
from typing import Any, Optional, Union

from SafeCounter import SafeCounter
from capture.ImprovedStreamCapture import ImprovedStreamCapture
# from capture.FFmpegCapture import FFmpegStreamCapture as ImprovedStreamCapture
from capture.MotionDetector import MotionDetection
from capture.utils import safe_put_queue, downscale_bgr, to_gray

# Centralized logger for all capture modules
from utils.logger import get_logger
logger = get_logger(__name__)


def capture_worker(
        rtsp_url: str,
        https_url: str,
        stream_id: str,
        target_fps: float,
        out_queue: Union[mp.Queue, Any],
        width: int,
        height: int,
        per_stream_cfg: dict,
        stop_event: Optional[Union[mp.Event, threading.Event]] = None,
):
    """
    Per-stream worker that captures frames, runs motion detection, and pushes frames to queue.
    Motion-gated sampling with pre-buffer, active mode, and clean shutdown via stop_event.

    Payload: (jpeg_bytes, timestamp, stream_id, frame_number)
    """

    logger.info(f"{stream_id}: worker started ({width}x{height})")

    cap = ImprovedStreamCapture(rtsp_url, https_url, width=width, height=height)
    frame_counter = SafeCounter()

    motion_detector = MotionDetection(
        min_area=per_stream_cfg.get("motion_min_area", 500),
        history_frames=per_stream_cfg.get("motion_history_frames", 15),
        threshold=per_stream_cfg.get("motion_threshold", 20)
    )

    pre_buffer = deque(maxlen=int(per_stream_cfg.get("pre_buffer_len", 3)))

    sensor_fps = float(per_stream_cfg.get("sensor_fps", 2.0))
    sensor_interval = 1.0 / max(0.1, sensor_fps)

    motion_inactive_timeout = float(per_stream_cfg.get("motion_inactive_timeout", 1.5))
    max_active_seconds = float(per_stream_cfg.get("max_active_seconds", 15.0))
    jpeg_quality = int(per_stream_cfg.get("jpeg_quality", 60))
    motion_w = int(per_stream_cfg.get("motion_w", 160))
    motion_h = int(per_stream_cfg.get("motion_h", 90))

    mode = "sensor"
    last_sensor_ts = 0.0
    motion_last_seen_ts = 0.0
    active_started_ts = 0.0

    should_stop = (lambda: False) if stop_event is None else stop_event.is_set

    try:
        while True:
            if should_stop():
                break

            now = time.time()
            if mode == "sensor" and (now - last_sensor_ts < sensor_interval):
                time.sleep(0.002)
                continue

            frame = cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            ts = now

            if mode == "sensor":
                last_sensor_ts = now
                small_bgr = downscale_bgr(frame, motion_w, motion_h)
                small_gray = to_gray(small_bgr)

                if motion_detector.detect(small_gray):
                    mode = "active"
                    motion_last_seen_ts = now
                    active_started_ts = now
                    logger.debug(f"{stream_id}: motion detected, switching to ACTIVE mode")

                    while pre_buffer:
                        ctx_frame, ctx_ts, ctx_idx = pre_buffer.popleft()
                        ok, buf = cv2.imencode(".jpg", ctx_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if ok:
                            payload = (buf.tobytes(), ctx_ts, stream_id, ctx_idx)
                            safe_put_queue(out_queue, payload)
                    continue
                else:
                    idx = frame_counter.increment()
                    pre_buffer.append((frame, ts, idx))
                continue

            # ACTIVE mode
            small_bgr = downscale_bgr(frame, motion_w, motion_h)
            small_gray = to_gray(small_bgr)
            if motion_detector.detect(small_gray):
                motion_last_seen_ts = now

            if (now - motion_last_seen_ts) > motion_inactive_timeout or (now - active_started_ts) >= max_active_seconds:
                mode = 'sensor'
                pre_buffer.clear()
                logger.debug(f"{stream_id}: switching back to SENSOR mode")
                continue

            idx = frame_counter.increment()
            if (idx % 2) != 0:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ok:
                    payload = (buf.tobytes(), ts, stream_id, idx)
                    safe_put_queue(out_queue, payload)

            time.sleep(0.001)

    except Exception as e:
        logger.exception(f"{stream_id}: unexpected error in capture_worker: {e}")

    finally:
        try:
            cap.release()
        except Exception as e:
            logger.exception(f"{stream_id}: error during release: {e}")
        logger.info(f"{stream_id}: worker stopped.")
