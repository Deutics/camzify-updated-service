# utils/service_helpers.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.services.filtering.object_size_filter import filter_detections_by_size

def fair_collect_batch(frame_queues, rr_counter, rr_keys, per_stream_cap, tick_timeout):
    """Fairly collect frames from all stream queues."""
    if not rr_keys:
        rr_keys = list(frame_queues.keys())

    n = len(rr_keys)
    if n == 0:
        return [], {}

    current_rotation = rr_counter.value
    pos = current_rotation % n
    frames_with_stream_id = []
    per_stream_counts = {}

    for i in range(n):
        sid = rr_keys[(pos + i) % n]
        q = frame_queues[sid]
        cnt = 0

        try:
            item = q.get(timeout=tick_timeout)
            if item:
                frames_with_stream_id.append((sid, item))
                cnt += 1
        except Exception:
            pass

        while cnt < per_stream_cap:
            try:
                nxt = q.get_nowait()
                if nxt:
                    frames_with_stream_id.append((sid, nxt))
                    cnt += 1
            except Exception:
                break

        if cnt > 0:
            per_stream_counts[sid] = cnt

    rr_counter.increment()
    return frames_with_stream_id, per_stream_counts


def parallel_filter_detections(detections_by_sid, thresholds_by_sid, logger):
    """Run per-stream objectdetectors filtering in parallel."""
    filtered = {}
    if not detections_by_sid:
        return filtered

    max_workers = min(4, len(detections_by_sid))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(filter_detections_by_size, detections, thresholds_by_sid.get(sid), "bbox"): sid
            for sid, detections in detections_by_sid.items()
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                filtered[sid] = fut.result()
            except Exception as e:
                logger.debug(f"Filter failed for {sid}: {e}")
                filtered[sid] = detections_by_sid[sid]
    return filtered
