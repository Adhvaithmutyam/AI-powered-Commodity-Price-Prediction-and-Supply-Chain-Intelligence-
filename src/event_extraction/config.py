"""
Configuration module for the supply-chain event extraction pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EventConfig:
    """Configuration settings for event extraction pipeline."""

    # Default input path (enriched sentiment file)
    input_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_news_sentiment.jsonl")
    )

    # Fallback input path (raw news file) if sentiment enrichment was not yet executed
    fallback_input_path: Path = field(
        default_factory=lambda: Path("data/raw/commodity_news.jsonl")
    )

    # Destination output path
    output_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_events.jsonl")
    )
