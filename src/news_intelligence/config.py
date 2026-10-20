"""
Configuration module for the News Intelligence ingestion pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PipelineConfig:
    """Configuration settings for GDELT news ingestion and relevance filtering."""

    # Target commodities to collect news for
    commodities: List[str] = field(
        default_factory=lambda: ["copper", "gold", "crude oil"]
    )

    # Search queries tailored per commodity
    # Targeted queries use domain operational terms, negative keywords (e.g. -sports),
    # exact phrases, and sourcelang:english to eliminate noise and non-English homonyms.
    commodity_queries: Dict[str, str] = field(
        default_factory=lambda: {
            "copper": 'copper (mining OR mine OR smelter OR cathode OR LME OR inventories OR strike OR "copper price") sourcelang:english',
            "gold": 'gold ("gold price" OR bullion OR "central bank" OR reserves OR "gold mining" OR "gold futures") -olympic -"gold medal" -sports sourcelang:english',
            "crude oil": '"crude oil" (OPEC OR production OR barrels OR refinery OR inventories OR EIA OR sanctions OR "oil price") sourcelang:english',
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
    rate_limit_delay_seconds: float = 6.0  # GDELT enforces at least 5s between requests
    max_retries: int = 3
    retry_backoff_seconds: float = 8.0
    user_agent: str = "CommodityNewsIntelligence/1.0 (Python/Requests)"
