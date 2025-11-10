import threading
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from src.utils.logger import get_logger  #  centralized logger import


class RuleConfigStore:
    """
    Thread-safe in-memory unified config store for both:
      - Rule configs (per-stream features/rules)
      - Stream configs (rtsp/https, dims, motion params)
    """

    def __init__(self, initial_rules: Dict[str, Any] = None, initial_streams: Optional[List[Dict[str, Any]]] = None):
        self._lock = threading.Lock()
        self._rules_by_id: Dict[str, Any] = initial_rules or {}
        self._streams_by_id: Dict[str, Dict[str, Any]] = {}
        if initial_streams:
            self.attach_streams(initial_streams)

        # Fine-grained subscribers
        self._on_stream_added: List[Callable[[str, Dict[str, Any]], None]] = []
        self._on_stream_removed: List[Callable[[str], None]] = []
        self._on_stream_updated: List[Callable[[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]], None]] = []
        self._on_rules_updated: List[Callable[[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]], None]] = []

        # Versioning
        self._version = int(time.time())

        # Logger
        self.logger = get_logger(self.__class__.__name__)

    # --------------------------
    # --- Stream access (new) ---
    # --------------------------
    def attach_streams(self, streams: List[Dict[str, Any]]):
        with self._lock:
            self._streams_by_id = {str(s["stream_id"]): dict(s) for s in (streams or [])}
            self._version = int(time.time())

    def get_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            s = self._streams_by_id.get(str(stream_id))
            return dict(s) if s is not None else None

    def get_all_streams(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {sid: dict(cfg) for sid, cfg in self._streams_by_id.items()}

    # ----------------------------------
    # --- Core rules access (existing) ---
    # ----------------------------------
    def get_active_features(self, stream_id: str) -> Dict[str, List[Dict]]:
        with self._lock:
            return self._rules_by_id.get(str(stream_id), {})

    def get_instance_id(self, stream_id: str, feature_name: str = "line_intrusion", index: int = 0) -> Optional[str]:
        with self._lock:
            stream_cfg = self._rules_by_id.get(str(stream_id), {})
            feature_list = stream_cfg.get(feature_name, [])
            if len(feature_list) > index:
                return feature_list[index].get("instance_id")
        return None

    # ------------------------------------------------
    # --- Fine-grained subscription registration   ---
    # ------------------------------------------------
    def on_stream_added(self, cb: Callable[[str, Dict[str, Any]], None]):
        with self._lock:
            self._on_stream_added.append(cb)

    def on_stream_removed(self, cb: Callable[[str], None]):
        with self._lock:
            self._on_stream_removed.append(cb)

    def on_stream_updated(self, cb: Callable[[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]], None]):
        with self._lock:
            self._on_stream_updated.append(cb)

    def on_rules_updated(self, cb: Callable[[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]], None]):
        with self._lock:
            self._on_rules_updated.append(cb)

    # -------------------------------------
    # --- Config Management (rules only) ---
    # -------------------------------------
    def refresh(self, new_rules_config: Dict[str, Any]):
        with self._lock:
            old_rules = self._rules_by_id
            self._rules_by_id = new_rules_config or {}
            self._version = int(time.time())
            fg_rules_subs = list(self._on_rules_updated)

            changed_streams = set(old_rules.keys()) | set(self._rules_by_id.keys())
            diffs: Dict[str, Dict[str, Any]] = {}
            for sid in changed_streams:
                oc = old_rules.get(sid, {})
                nc = self._rules_by_id.get(sid, {})
                if oc != nc:
                    ok = self._keys(oc)
                    nk = self._keys(nc)
                    changed_keys = sorted(list((ok ^ nk) | {k for k in ok & nk if oc.get(k) != nc.get(k)}))
                    diffs[sid] = {"changed_keys": changed_keys}

        self.logger.info(f"Rules refreshed (version={self._version})")
        self.logger.debug(f"Rules diffs for streams: {list(diffs.keys())}")

        for sid, diff in diffs.items():
            old_cfg = old_rules.get(sid, {})
            new_cfg = self._rules_by_id.get(sid, {})
            for cb in fg_rules_subs:
                try:
                    self.logger.debug(f"on_rules_updated({sid}) diff_keys={diff.get('changed_keys')}")
                    cb(sid, old_cfg, new_cfg, diff)
                except Exception as e:
                    self.logger.warning(f"on_rules_updated callback for {sid} raised: {e}", exc_info=True)

    def reload(self, new_rules_config: Dict[str, Any]):
        self.refresh(new_rules_config)

    # ---------------------------------------
    # --- Atomic mutation (streams + rules) ---
    # ---------------------------------------
    def apply_mutation(self, mutate_fn: Callable[[Dict[str, Dict[str, Any]], Dict[str, Any]], None]):
        with self._lock:
            old_streams = {sid: dict(cfg) for sid, cfg in self._streams_by_id.items()}
            old_rules = {sid: dict(cfg) for sid, cfg in self._rules_by_id.items()}

            mutate_fn(self._streams_by_id, self._rules_by_id)
            self._version = int(time.time())

            added, removed, updated, stream_diffs = self._diff_streams(old_streams, self._streams_by_id)
            rules_changed, rules_diffs = self._diff_rules(old_rules, self._rules_by_id)

            stream_added_subs = list(self._on_stream_added)
            stream_removed_subs = list(self._on_stream_removed)
            stream_updated_subs = list(self._on_stream_updated)
            rules_updated_subs = list(self._on_rules_updated)

        self.logger.info(
            f"Mutation: added={added}, removed={removed}, updated={updated}, rules_changed={rules_changed}"
        )

        for sid in removed:
            for cb in stream_removed_subs:
                try:
                    cb(sid)
                    self.logger.debug(f"on_stream_removed({sid})")
                except Exception as e:
                    self.logger.warning(f"on_stream_removed callback for {sid} failed: {e}", exc_info=True)

        for sid in added:
            new_cfg = self._streams_by_id.get(sid, {})
            for cb in stream_added_subs:
                try:
                    cb(sid, dict(new_cfg))
                    self.logger.debug(f"on_stream_added({sid})")
                except Exception as e:
                    self.logger.warning(f"on_stream_added callback for {sid} failed: {e}", exc_info=True)

        for sid in updated:
            old_cfg = old_streams.get(sid, {})
            new_cfg = self._streams_by_id.get(sid, {})
            diff = stream_diffs.get(sid, {"changed_keys": []})
            for cb in stream_updated_subs:
                try:
                    cb(sid, dict(old_cfg), dict(new_cfg), diff)
                    self.logger.debug(f"on_stream_updated({sid}) diff_keys={diff.get('changed_keys')}")
                except Exception as e:
                    self.logger.warning(f"on_stream_updated callback for {sid} failed: {e}", exc_info=True)

        for sid in rules_changed:
            old_cfg = old_rules.get(sid, {})
            new_cfg = self._rules_by_id.get(sid, {})
            diff = rules_diffs.get(sid, {"changed_keys": []})
            for cb in rules_updated_subs:
                try:
                    cb(sid, dict(old_cfg), dict(new_cfg), diff)
                    self.logger.debug(f"on_rules_updated({sid}) diff_keys={diff.get('changed_keys')}")
                except Exception as e:
                    self.logger.warning(f"on_rules_updated callback for {sid} failed: {e}", exc_info=True)

    # --------------------------
    # --- Version Management ---
    # --------------------------
    def get_version(self) -> int:
        with self._lock:
            return self._version

    def get_rules_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._rules_by_id)

    def get_streams_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._streams_by_id)

    # --------------------------
    # --- Internal diff utils ---
    # --------------------------
    @staticmethod
    def _keys(obj):
        return set(obj.keys()) if isinstance(obj, dict) else set()

    @staticmethod
    def _diff_streams(old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]):
        old_ids = set(old.keys())
        new_ids = set(new.keys())
        added = sorted(list(new_ids - old_ids))
        removed = sorted(list(old_ids - new_ids))
        updated = []
        diffs: Dict[str, Dict[str, Any]] = {}
        for sid in (old_ids & new_ids):
            o = old.get(sid, {})
            n = new.get(sid, {})
            if o != n:
                updated.append(sid)
                ok = RuleConfigStore._keys(o)
                nk = RuleConfigStore._keys(n)
                changed_keys = sorted(list((ok ^ nk) | {k for k in ok & nk if o.get(k) != n.get(k)}))
                diffs[sid] = {"changed_keys": changed_keys}
        return added, removed, updated, diffs

    @staticmethod
    def _diff_rules(old: Dict[str, Any], new: Dict[str, Any]):
        all_sids = set(old.keys()) | set(new.keys())
        changed_sids = []
        diffs: Dict[str, Dict[str, Any]] = {}
        for sid in sorted(all_sids):
            o = old.get(sid, {})
            n = new.get(sid, {})
            if o != n:
                changed_sids.append(sid)
                ok = RuleConfigStore._keys(o)
                nk = RuleConfigStore._keys(n)
                changed_keys = sorted(list((ok ^ nk) | {k for k in ok & nk if o.get(k) != n.get(k)}))
                diffs[sid] = {"changed_keys": changed_keys}
        return changed_sids, diffs
