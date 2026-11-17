"""
Configuration module for NLP Feature Engineering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FeatureConfig:
    """Configuration settings for NLP feature building and timestamp aggregation."""

    # Input file paths
    raw_news_path: Path = field(
        default_factory=lambda: Path("data/raw/commodity_news.jsonl")
    )
    sentiment_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_news_sentiment.jsonl")
    )
    events_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_events.jsonl")
    )

    # Output file paths
    article_features_path: Path = field(
        default_factory=lambda: Path("data/processed/nlp_article_features.jsonl")
    )
    aggregated_features_path: Path = field(
        default_factory=lambda: Path("data/processed/nlp_aggregated_features.jsonl")
    )

    # Target commodities
    commodities: List[str] = field(
        default_factory=lambda: ["copper", "gold", "crude oil"]
    )

    # Aggregation window sizes in days (e.g., 1 day and 5 days)
    aggregation_window_days: List[int] = field(
        default_factory=lambda: [1, 5]
    )

    # High-confidence threshold for events
    event_confidence_threshold: float = 0.70
