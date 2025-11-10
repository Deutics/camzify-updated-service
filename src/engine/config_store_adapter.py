# rule_engines/ConfigStoreAdapter.py
from typing import Dict, Any

class ConfigStoreAdapter:
    """
    Minimal adapter for a config store. Provides unified access to active features
    and optional refresh. Legacy per-stream subscribe is removed.
    """

    def __init__(self, raw_store: Any):
        self._store = raw_store

    def get_active_features(self, stream_id: str) -> Dict[str, Any]:
        return self._store.get_active_features(stream_id)

    def refresh(self, new_cfg: Dict[str, Any]) -> None:
        if hasattr(self._store, "refresh"):
            self._store.refresh(new_cfg)
