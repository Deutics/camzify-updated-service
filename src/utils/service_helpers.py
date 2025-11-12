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


def safe_join_thread(t, timeout=1.0):
    try:
        t.join(timeout=timeout)
    except Exception:
        pass

def safe_join_process(p, timeout=2.0):
    try:
        p.join(timeout=timeout)
    except Exception:
        pass
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
