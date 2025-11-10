"""
rule_engine package exports the top-level manager and helpers.
"""
from src.engine.rule_engine_manager import RuleEngineManager
from src.engine.factories import register_feature, get_feature_factory, FEATURE_REGISTRY

__all__ = ["RuleEngineManager", "register_feature", "get_feature_factory", "FEATURE_REGISTRY"]