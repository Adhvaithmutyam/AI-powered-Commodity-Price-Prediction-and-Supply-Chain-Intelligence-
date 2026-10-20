"""
News Intelligence module for commodity news ingestion, normalization, and persistence.
"""

from .config import PipelineConfig
from .gdelt_client import GDELTClient
from .normalizer import normalize_article
from .storage import load_existing_keys, append_records
from .pipeline import NewsIngestionPipeline

__all__ = [
    "PipelineConfig",
    "GDELTClient",
    "normalize_article",
    "load_existing_keys",
    "append_records",
    "NewsIngestionPipeline",
]
