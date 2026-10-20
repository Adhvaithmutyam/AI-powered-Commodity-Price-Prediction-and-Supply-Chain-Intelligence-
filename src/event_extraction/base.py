"""
Base classes and interfaces for supply-chain event extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


VALID_EVENT_TYPES = {
    "supply_disruption",
    "demand_change",
    "geopolitical_risk",
    "inventory_change",
    "policy_action",
    "production_change",
    "other",
}

VALID_IMPACT_DIRECTIONS = {
    "potentially_bullish",
    "potentially_bearish",
    "mixed",
    "unknown",
}


@dataclass
class EventRecord:
    """Standardized supply-chain event record schema."""

    commodity: str
    event_type: str
    event_summary: str
    impact_direction: str
    confidence: float
    evidence_url: str
    article_publication_timestamp: Optional[str]
    extraction_status: str
    evidence_fact: Optional[str] = None
    implication_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for JSON serialization."""
        return asdict(self)


class BaseEventExtractor(ABC):
    """Abstract base class for commodity news event extractors."""

    @abstractmethod
    def extract_event(self, record: Dict[str, Any]) -> EventRecord:
        """
        Extract a supply-chain event from a single article record.

        Args:
            record: Normalized article dictionary containing at least 'title' and 'url'.

        Returns:
            EventRecord with structured extraction fields.
        """
        pass

    def extract_events(self, records: List[Dict[str, Any]]) -> List[EventRecord]:
        """
        Extract events from a list of article records.

        Args:
            records: List of article dictionaries.

        Returns:
            List of EventRecord instances.
        """
        return [self.extract_event(rec) for rec in records if isinstance(rec, dict)]
