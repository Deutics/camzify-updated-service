import threading
from typing import Dict, Any, Optional, List, Callable
from src.utils.logger import get_logger


class RuleConfigStore:
    """
    Thread-safe in-memory store for unified stream + feature configuration.
    Each stream contains its metadata (URL, dimensions) and features (rules).
    """

    def __init__(self, initial_streams: Dict[str, Any]):
        self._lock = threading.Lock()
        self._streams_by_id: Dict[str, Dict[str, Any]] = dict(initial_streams or {})
        self._version = 0
        self.logger = get_logger(self.__class__.__name__)

        # Subscribers
        self._on_stream_added: List[Callable[[str, Dict[str, Any]], None]] = []
        self._on_stream_removed: List[Callable[[str], None]] = []
        self._on_stream_updated: List[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = []
        self._on_rules_updated: List[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = []

        self._version += 1
        self.logger.info(f"RuleConfigStore initialized with {len(self._streams_by_id)} streams")

    # --------------------------
    # --- Stream access ---
    # --------------------------
    def get_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cfg = self._streams_by_id.get(str(stream_id))
            return dict(cfg) if cfg else None

    def get_all_streams(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {sid: dict(cfg) for sid, cfg in self._streams_by_id.items()}

    # --------------------------
    # --- Feature / rules access ---
    # --------------------------
    def get_active_features(self, stream_id: str) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            stream_cfg = self._streams_by_id.get(str(stream_id), {})
            return stream_cfg.get("features", {})

    def get_instance_id(self, stream_id: str, feature_name: str = "line_intrusion", index: int = 0) -> Optional[str]:
        features = self.get_active_features(stream_id).get(feature_name, [])
        if index < len(features):
            return features[index].get("instance_id")
        return None

    # --------------------------
    # --- Update / refresh ---
    # --------------------------
    def refresh(self, new_streams: Dict[str, Any]):
        with self._lock:
            old_streams = self._streams_by_id
            self._streams_by_id = dict(new_streams or {})
            self._version += 1

        # Identify changes
        old_keys = set(old_streams.keys())
        new_keys = set(self._streams_by_id.keys())

        added = new_keys - old_keys
        removed = old_keys - new_keys
        updated = {sid for sid in new_keys & old_keys if old_streams[sid] != self._streams_by_id[sid]}

        # Trigger callbacks
        for sid in added:
            for cb in self._on_stream_added:
                try:
                    cb(sid, dict(self._streams_by_id[sid]))
                except Exception as e:
                    self.logger.warning(f"on_stream_added callback failed for {sid}: {e}")

        for sid in removed:
            for cb in self._on_stream_removed:
                try:
                    cb(sid)
                except Exception as e:
                    self.logger.warning(f"on_stream_removed callback failed for {sid}: {e}")

        for sid in updated:
            for cb in self._on_stream_updated:
                try:
                    cb(sid, dict(old_streams[sid]), dict(self._streams_by_id[sid]))
                except Exception as e:
                    self.logger.warning(f"on_stream_updated callback failed for {sid}: {e}")

        self.logger.info(f"RuleConfigStore refreshed: added={len(added)}, removed={len(removed)}, updated={len(updated)}")

    # --------------------------
    # --- Subscription registration ---
    # --------------------------
    def on_stream_added(self, cb: Callable[[str, Dict[str, Any]], None]):
        with self._lock:
            self._on_stream_added.append(cb)

    def on_stream_removed(self, cb: Callable[[str], None]):
        with self._lock:
            self._on_stream_removed.append(cb)

    def on_stream_updated(self, cb: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        with self._lock:
            self._on_stream_updated.append(cb)

    def on_rules_updated(self, cb: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        with self._lock:
            self._on_rules_updated.append(cb)

    # --------------------------
    # --- Version / snapshot ---
    # --------------------------
    def get_version(self) -> int:
        with self._lock:
            return self._version

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._streams_by_id)
