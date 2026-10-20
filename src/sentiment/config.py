"""
Configuration module for the FinBERT Sentiment Analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class SentimentConfig:
    """Configuration for sentiment inference and storage."""

    # Pretrained financial sentiment model
    model_name: str = "ProsusAI/finbert"

    # Input and output file paths
    input_path: Path = field(
        default_factory=lambda: Path("data/raw/commodity_news.jsonl")
    )
    output_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_news_sentiment.jsonl")
    )

    # Inference settings
    batch_size: int = 16
    max_length: int = 512
    truncation: bool = True

    # Compute device: auto-detects GPU if available, defaults to CPU
    device: Optional[str] = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
