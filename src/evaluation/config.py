"""
Configuration module for the NLP Evaluation Framework and Data-Quality Validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


# Standard Taxonomies aligned with existing pipeline schemas
VALID_COMMODITIES: Set[str] = {"copper", "gold", "crude oil"}

VALID_RELEVANCE_CLASSES: Set[str] = {
    "directly_relevant",
    "potentially_relevant",
    "irrelevant",
    "uncertain",
}

VALID_SENTIMENT_LABELS: Set[str] = {
    "positive",
    "negative",
    "neutral",
}

VALID_EVENT_TYPES: Set[str] = {
    "supply_disruption",
    "demand_change",
    "geopolitical_risk",
    "inventory_change",
    "policy_action",
    "production_change",
    "other",
}

VALID_IMPACT_DIRECTIONS: Set[str] = {
    "potentially_bullish",
    "potentially_bearish",
    "mixed",
    "unknown",
}

VALID_FEATURE_STATUSES: Set[str] = {
    "complete",
    "partial_missing_events",
    "partial_missing_sentiment",
    "partial_missing_both",
}


@dataclass
class EvaluationConfig:
    """Configuration settings for NLP data validation, annotation, and evaluation."""

    # Default file paths
    article_features_path: Path = field(
        default_factory=lambda: Path("data/processed/nlp_article_features.jsonl")
    )
    raw_news_path: Path = field(
        default_factory=lambda: Path("data/raw/commodity_news.jsonl")
    )
    labeled_dataset_path: Optional[Path] = None

    # Default report and output paths
    data_quality_report_path: Path = field(
        default_factory=lambda: Path("data/evaluation/data_quality_report.json")
    )
    annotation_sample_path: Path = field(
        default_factory=lambda: Path("data/evaluation/annotation_sample.jsonl")
    )
    evaluation_report_path: Path = field(
        default_factory=lambda: Path("data/evaluation/classification_report.json")
    )

    # Target commodities
    commodities: List[str] = field(
        default_factory=lambda: ["copper", "gold", "crude oil"]
    )

    # Confidence validation bounds
    confidence_min: float = 0.0
    confidence_max: float = 1.0

    # Annotation and sample generation settings
    default_sample_size: int = 50
    default_seed: int = 42

    # Minimum sample threshold for reliable supervised evaluation
    min_labeled_sample_size: int = 30
