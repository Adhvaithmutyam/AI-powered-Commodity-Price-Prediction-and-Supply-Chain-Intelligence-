"""
News Intelligence module for commodity news ingestion, normalization, persistence, and relevance assessment.
"""

from .config import PipelineConfig
from .gdelt_client import GDELTClient
from .normalizer import normalize_article
from .storage import load_existing_keys, append_records
from .pipeline import NewsIngestionPipeline
from .relevance import (
    RelevanceClass,
    RelevanceResult,
    RelevanceClassifier,
    assess_relevance,
    enrich_record_with_relevance,
)

__all__ = [
    "PipelineConfig",
    "GDELTClient",
    "normalize_article",
    "load_existing_keys",
    "append_records",
    "NewsIngestionPipeline",
    "RelevanceClass",
    "RelevanceResult",
    "RelevanceClassifier",
    "assess_relevance",
    "enrich_record_with_relevance",
]
