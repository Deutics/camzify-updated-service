from datetime import datetime, time as dtime
from typing import Dict, Any


def is_stream_active(cfg: Dict[str, Any]) -> bool:
    """Check if stream is active. Currently always returns True."""
    return cfg.get("is_active", True)


def within_time_bounds(cfg: Dict[str, Any], timestamp: float) -> bool:
    """Check if timestamp falls within configured time bounds."""
    frame_time = datetime.fromtimestamp(timestamp).time()
    try:
        start_time = dtime.fromisoformat(cfg.get("time_bound_start", "00:00:00"))
        end_time = dtime.fromisoformat(cfg.get("time_bound_end", "23:59:59"))
        if start_time <= end_time:
            return start_time <= frame_time <= end_time
        else:
            return frame_time >= start_time or frame_time <= end_time
    except Exception as e:
        print(f"[WARNING] Time bound parsing error for stream {cfg.get('stream_id', 'unknown')}: {e}")
        return True