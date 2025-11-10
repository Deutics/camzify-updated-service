# pipelines/ForwarderWorker.py
import time
import cv2
import numpy as np
import queue as _queue
from typing import Dict, Any, Callable
from data_models.FrameData import FrameData
from SafeCounter import SafeCounter
from utils.logger import get_logger

logger = get_logger(__name__)


class ForwarderWorker:
    """
    Reads JPEG bytes from multiprocess input queue, decodes to BGR frames,
    applies early filters (active status, time bounds), wraps into FrameData,
    and pushes to local per-stream queue.
    """

    def __init__(
            self,
            stream_id: str,
            in_queue,
            out_queue,
            stream_cfg: Dict[str, Any],
            is_active_fn: Callable[[Dict[str, Any]], bool],
            within_time_bounds_fn: Callable[[Dict[str, Any], float], bool],
            stop_event,
            instance_id,
    ):
        self.stream_id = stream_id
        self.in_q = in_queue
        self.out_q = out_queue
        self.cfg = stream_cfg
        self.is_active = is_active_fn
        self.within_time_bounds = within_time_bounds_fn
        self._stop_event = stop_event
        self.instance_id = instance_id

        # Centralized logger for consistent service logging

    def run(self):
        """Main loop: decode, filter, forward with adaptive thinning."""
        last_stats_time = time.time()

        # Protected counters for stats (reset every second)
        frames_forwarded_count = 0
        frames_dropped_count = 0
        frames_filtered_inactive_count = 0
        frames_filtered_time_count = 0
        frames_thinned_count = 0

        # Protected counter for adaptive thinning (never reset, unbounded)
        frames_processed_counter = SafeCounter()

        # Adaptive thinning params
        thin_threshold = 0.70  # start thinning when queue >70% full
        thin_modulo = 2  # drop every 2nd frame when thinning active

        while not self._stop_event.is_set():
            try:
                payload = self.in_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"[STREAM {self.stream_id}] input queue get error: {e}", exc_info=True)
                time.sleep(0.05)
                continue

            try:
                # payload: (jpeg_bytes, timestamp, stream_id, frame_number)
                jpeg_bytes, ts, sid, frame_num = payload
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    frames_dropped_count += 1
                    continue

                # Filter 1: stream active status
                if not self.is_active(self.cfg):
                    frames_filtered_inactive_count += 1
                    continue

                # Filter 2: time bounds
                if not self.within_time_bounds(self.cfg, ts if ts else time.time()):
                    frames_filtered_time_count += 1
                    continue

                # Adaptive thinning: only drop frames when output queue is filling up
                current_frame_index = frames_processed_counter.increment()
                try:
                    qsize = self.out_q.qsize()
                    maxsize = getattr(self.out_q, "maxsize", 64) or 64
                    fill_ratio = qsize / max(1, maxsize)
                except Exception:
                    fill_ratio = 0.0  # if qsize() not available, assume not full

                if fill_ratio > thin_threshold and (current_frame_index % thin_modulo) != 0:
                    # Queue is filling; drop this frame to prevent backpressure
                    frames_thinned_count += 1
                    continue

                fd = FrameData(
                    frame=frame,
                    timestamp=ts if ts else time.time(),
                    stream_id=sid or self.stream_id,
                    frame_number=frame_num or 0,
                    instance_id=self.instance_id
                )

                if self.out_q.put(fd):
                    frames_forwarded_count += 1
                else:
                    frames_dropped_count += 1

            except Exception as e:
                logger.error(f"[ERROR] Frame processing error in stream {self.stream_id}: {e}", exc_info=True)
                frames_dropped_count += 1

            now = time.time()
            if now - last_stats_time >= 1.0:
                logger.info(
                    f"[STREAM {self.stream_id}] Forwarded:{frames_forwarded_count} Dropped:{frames_dropped_count} "
                    f"Thinned:{frames_thinned_count} Filtered(inactive):{frames_filtered_inactive_count} "
                    f"Filtered(time):{frames_filtered_time_count}"
                )
                frames_forwarded_count = frames_dropped_count = frames_filtered_inactive_count = frames_filtered_time_count = frames_thinned_count = 0
                last_stats_time = now
