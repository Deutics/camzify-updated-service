# rule_engine/factories.py

from typing import Dict, Optional, Callable,Any
from features.ImprovedLineIntrusionDetector import ImprovedLineIntrusionDetector

FEATURE_REGISTRY = {}

def register_feature(name: str, factory: Callable[[Dict[str, Any], str, Any, Any], Any]) -> None:
    FEATURE_REGISTRY[name] = factory

def get_feature_factory(name: str) -> Optional[Callable[[Dict[str, Any], str, Any, Any], Any]]:
    return FEATURE_REGISTRY.get(name)

# Register refactored feature directly (NO adapter)
register_feature("line_intrusion", lambda cfg, sid, alert_sys, viz: ImprovedLineIntrusionDetector(cfg, sid, alert_sys, viz))

# from WeaponDetectionFeature import WeaponDetectionFeature
#
# register_feature("weapon_detection", lambda cfg, sid, alert_sys, viz: WeaponDetectionFeature(cfg, sid, alert_sys, viz))
