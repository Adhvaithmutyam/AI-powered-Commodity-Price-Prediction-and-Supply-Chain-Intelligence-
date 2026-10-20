"""
Configuration module for the News Intelligence ingestion pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PipelineConfig:
    """Configuration settings for GDELT news ingestion."""

    # Target commodities to collect news for
    commodities: List[str] = field(
        default_factory=lambda: ["copper", "gold", "crude oil"]
    )

    # Search queries tailored per commodity
    # For multi-word phrases like 'crude oil', quotation marks ensure exact phrase matching in GDELT.
    commodity_queries: Dict[str, str] = field(
        default_factory=lambda: {
            "copper": "copper",
            "gold": "gold",
            "crude oil": '"crude oil"',
        }
    )

    # GDELT DOC 2.0 API endpoint
    api_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Default output storage file path
    output_path: Path = field(
        default_factory=lambda: Path("data/raw/commodity_news.jsonl")
    )

    # API request parameters
    max_records: int = 50
    timespan: str = "7d"  # Default: last 7 days of news
    sort: str = "datedesc"
    mode: str = "artlist"
    output_format: str = "json"

    # Network, retry, and rate-limiting settings
    timeout_seconds: int = 30
    rate_limit_delay_seconds: float = 5.5  # GDELT requires at least 5s between requests
    max_retries: int = 3
    retry_backoff_seconds: float = 6.0
    user_agent: str = "CommodityNewsIntelligence/1.0 (Python/Requests)"
