# rule_engine/utils.py
from typing import List, Dict, Any

from rule_engine import FeatureBase


def bind_store_callback_safely(store, event_name: str, handler):
    """
    Wrap a RuleConfigStore event subscription in a safe try/catch,
    preventing exceptions in callbacks from breaking service.
    """
    try:
        event = getattr(store, event_name, None)
        if callable(event):
            event(handler)
    except Exception:
        pass

def safe_call(func, *args, default=None, **kwargs):
    """
    Safely call a function, catching exceptions.
    Returns `default` on failure.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        from utils.logger import get_logger
        logger = get_logger("rule_engine_utils")
        logger.exception(f"safe_call failed: {e}")
        return default

def extract_overlay_geometry(feature_instances: List[FeatureBase]) -> List[Dict[str, Any]]:
    overlay = []
    for inst in feature_instances:
        lp = getattr(inst, "line_points", None)
        if lp:
            overlay.append({"type": "line", "points": lp})
        zp = getattr(inst, "zone_points", None)
        if zp:
            overlay.append({"type": "zone", "points": zp})
    return overlay
