"""
Supply-chain event extraction and market intelligence module.
"""

from .base import (
    BaseEventExtractor,
    EventRecord,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
)
from .config import EventConfig
from .patterns import EVENT_PATTERNS
from .rule_based import RuleBasedEventExtractor
from .pipeline import EventExtractionPipeline

__all__ = [
    "BaseEventExtractor",
    "EventRecord",
    "EventConfig",
    "EVENT_PATTERNS",
    "RuleBasedEventExtractor",
    "EventExtractionPipeline",
    "VALID_EVENT_TYPES",
    "VALID_IMPACT_DIRECTIONS",
]
