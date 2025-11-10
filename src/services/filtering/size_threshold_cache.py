import threading
from typing import Optional, Dict, Any, Callable
from src.services.filtering.object_size_filter import compute_size_thresholds
from src.utils.logger import get_logger

# Centralized logger consistent with service-wide logging
logger = get_logger(__name__)


class SizeThresholdCache:
    """
    Thread-safe cache for per-stream object size thresholds.
    Automatically invalidates on stream or config updates.
    """

    def __init__(self, config_store, get_stream_dims_fn, default_frame_size=(640, 480), debug: bool = False):
        self._store = config_store
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._get_stream_dims = get_stream_dims_fn  # callable: sid -> (w,h) or None
        self._default_w, self._default_h = default_frame_size
        self.debug = debug

        if self.debug:
            logger.setLevel("DEBUG")

        # Hook into RuleConfigStore fine-grained events
        self._bind_store_events_if_available()

    # ----------------------------------------------------------------------
    # Event bindings
    # ----------------------------------------------------------------------
    def _bind_store_events_if_available(self):
        """Attach to the fine-grained RuleConfigStore events."""
        on_stream_updated = getattr(self._store, "on_stream_updated", None)
        on_stream_removed = getattr(self._store, "on_stream_removed", None)
        on_rules_updated = getattr(self._store, "on_rules_updated", None)

        # Invalidate when stream dimensions or rule config changes
        if on_stream_updated:
            def _on_stream_updated_cb(stream_id: str, old_cfg: Dict[str, Any],
                                      new_cfg: Dict[str, Any], diff: Dict[str, Any]):
                try:
                    changed = set(diff.get("changed_keys", []))
                    if {"camera_width", "camera_height"} & changed:
                        self.invalidate(stream_id)
                        logger.info(f"Invalidated (dims change) for stream {stream_id}")
                except Exception as e:
                    logger.warning(f"Error handling stream update for {stream_id}: {e}", exc_info=True)
            on_stream_updated(_on_stream_updated_cb)

        # Invalidate when rules change
        if on_rules_updated:
            def _on_rules_updated_cb(stream_id: str, old_rules: Dict[str, Any],
                                     new_rules: Dict[str, Any], diff: Dict[str, Any]):
                try:
                    self.invalidate(stream_id)
                    logger.info(f"Invalidated (rules change) for stream {stream_id}")
                except Exception as e:
                    logger.warning(f"Error handling rules update for {stream_id}: {e}", exc_info=True)
            on_rules_updated(_on_rules_updated_cb)

        # Remove from cache when stream removed
        if on_stream_removed:
            def _on_stream_removed_cb(stream_id: str):
                try:
                    self.detach_stream(stream_id)
                    logger.info(f"Purged {stream_id} on removal")
                except Exception as e:
                    logger.warning(f"Error purging cache for {stream_id}: {e}", exc_info=True)
            on_stream_removed(_on_stream_removed_cb)

    # ----------------------------------------------------------------------
    # Core cache API
    # ----------------------------------------------------------------------
    def invalidate(self, stream_id: str):
        """Manually invalidate cache for a stream."""
        with self._lock:
            self._cache.pop(stream_id, None)
            if self.debug:
                logger.debug(f"Invalidated cache entry for {stream_id}")

    def detach_stream(self, stream_id: str):
        """Alias for invalidate, for MSVAS compatibility."""
        self.invalidate(stream_id)

    def attach_to_streams(self, stream_ids):
        """
        Optional initializer hook to pre-compute or confirm active streams.
        Currently a no-op except for clearing stale entries.
        """
        with self._lock:
            for sid in stream_ids:
                if sid not in self._cache:
                    self._cache[sid] = None
                    if self.debug:
                        logger.debug(f"Attached empty cache slot for {sid}")

    def detach(self):
        """Clear all cached thresholds."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared all cached thresholds")

    # ----------------------------------------------------------------------
    # Retrieval + recomputation
    # ----------------------------------------------------------------------
    def get(self, sid: str):
        """Return thresholds for the given stream, recomputing if needed."""
        with self._lock:
            if sid in self._cache and self._cache[sid] is not None:
                if self.debug:
                    logger.debug(f"Cache hit for {sid}")
                return self._cache[sid]

        try:
            active = self._store.get_active_features(sid) or {}
            line_rules = active.get("line_intrusion", [])
            rc = line_rules[0] if line_rules else {}

            # Get stream dimensions
            try:
                dims = self._get_stream_dims(sid)
            except Exception:
                dims = None
            sw, sh = (dims if dims and len(dims) == 2 else (None, None))

            cam_w = int(rc.get("camera_width", 0)) or (sw or self._default_w)
            cam_h = int(rc.get("camera_height", 0)) or (sh or self._default_h)

            bb_start = rc.get("bounding_box_start", (0, 0))
            bb_end = rc.get("bounding_box_end", (cam_w, cam_h))
            precision = rc.get("bounding_box_precision_factor", rc.get("precision_factor", None))

            th = compute_size_thresholds(bb_start, bb_end, precision)
            logger.debug(f"Recomputed thresholds for stream {sid}")

        except Exception as e:
            logger.error(f"Failed to compute thresholds for stream {sid}: {e}", exc_info=True)
            th = None

        with self._lock:
            self._cache[sid] = th
        return th
