from datetime import datetime, time as dtime
from typing import Dict, Any


def is_stream_active(rule_config_store, stream_id: str) -> bool:
    """
    Return True if the stream has at least one active feature.
    rule_config_store knows which features are enabled per stream.
    """
    active_features = rule_config_store.get_active_features(stream_id)
    return bool(active_features)  # True if at least one feature is active




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