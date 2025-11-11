# rule_engine/FeatureBase.py

"""
Defines the canonical feature interface used by the StreamRuleEngine.
All feature implementations (and adapters) should inherit or follow this contract.
"""

from typing import List, Optional, Dict
from src.data_models.frame_data import FrameData
from src.data_models.tracked_object import TrackedObject


class FeatureBase:
    """
    Base interface for a feature.

    Implementations should expose:
      - check_intrusion(tracked_objects: List[TrackedObject], frame_data: Optional[FrameData], meta: Dict) -> List[dict]
      - update_config(new_cfg: Dict) -> None
      - shutdown() -> None

    NOTE: not using abc.ABC to keep adapters lightweight. This class documents the expected shape.
    """

    def check_intrusion(self, tracked_objects: List[TrackedObject], frame_data: Optional[FrameData], meta: Dict) -> List[dict]:
        raise NotImplementedError

    def update_config(self, new_cfg: Dict) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError