"""
FinBERT sentiment analysis module for financial news headline classification.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import torch

from .config import SentimentConfig

logger = logging.getLogger(__name__)


class SentimentAnalyzerError(Exception):
    """Base exception for sentiment analyzer errors."""


class ModelLoadError(SentimentAnalyzerError):
    """Raised when the sentiment classification model fails to load."""


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analyzer using ProsusAI/finbert via Hugging Face Transformers.
    
    Supports:
    - Lazy loading (model downloaded/loaded only on first inference)
    - Positive, negative, and neutral classifications with confidence scores
    - Safe handling for missing or blank titles
    - Device autodetection (GPU vs CPU)
    - Batched inference for optimal throughput
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        pipeline_instance: Optional[Any] = None,
    ):
        self.config = config or SentimentConfig()
        self._pipeline = pipeline_instance

    def _resolve_device(self) -> Union[int, str]:
        """Convert device string to Hugging Face pipeline device parameter."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return 0  # GPU index 0
        return -1  # CPU

    def _create_pipeline(self) -> Any:
        """Create the Hugging Face sentiment-analysis pipeline."""
        from transformers import pipeline

        device_param = self._resolve_device()
        return pipeline(
            task="sentiment-analysis",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            device=device_param,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
        )

    def load_model(self) -> None:
        """Explicitly load the pretrained model and tokenizer pipeline."""
        if self._pipeline is not None:
            return

        logger.info("Loading sentiment model: %s", self.config.model_name)
        try:
            self._pipeline = self._create_pipeline()
            logger.info("FinBERT model loaded successfully on %s", self.config.device)
        except Exception as e:
            logger.error("Failed to load sentiment model '%s': %s", self.config.model_name, e)
            raise ModelLoadError(f"Could not initialize model '{self.config.model_name}': {e}") from e

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying model pipeline is loaded."""
        return self._pipeline is not None

    def analyze_text(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Analyze sentiment for a single text headline.

        Returns a dictionary containing:
            - sentiment_label: 'positive', 'negative', 'neutral', or 'unavailable'
            - sentiment_score: float confidence score or None
            - sentiment_model: model name string
            - sentiment_status: 'success', 'empty_text', or error details
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "sentiment_label": "unavailable",
                "sentiment_score": None,
                "sentiment_model": self.config.model_name,
                "sentiment_status": "empty_text",
            }

        cleaned_text = text.strip()

        try:
            self.load_model()
            results = self._pipeline(cleaned_text)
            if not results:
                return {
                    "sentiment_label": "unavailable",
                    "sentiment_score": None,
                    "sentiment_model": self.config.model_name,
                    "sentiment_status": "empty_result",
                }

            top_result = results[0] if isinstance(results, list) else results
            label = str(top_result.get("label", "unavailable")).strip().lower()
            score = round(float(top_result.get("score", 0.0)), 4)

            return {
                "sentiment_label": label,
                "sentiment_score": score,
                "sentiment_model": self.config.model_name,
                "sentiment_status": "success",
            }
        except Exception as e:
            logger.warning("Sentiment inference failed for text '%s': %s", cleaned_text[:50], e)
            return {
                "sentiment_label": "unavailable",
                "sentiment_score": None,
                "sentiment_model": self.config.model_name,
                "sentiment_status": f"error: {str(e)}",
            }

    def analyze_batch(self, texts: List[Optional[str]]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of texts efficiently.
        Handles empty or invalid strings safely without sending them to the model.
        """
        if not texts:
            return []

        # Prepare outputs initialized to empty/fallback
        outputs: List[Dict[str, Any]] = [
            {
                "sentiment_label": "unavailable",
                "sentiment_score": None,
                "sentiment_model": self.config.model_name,
                "sentiment_status": "empty_text",
            }
            for _ in texts
        ]

        # Identify valid non-empty indices and their texts
        valid_indices: List[int] = []
        valid_texts: List[str] = []
        for idx, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text.strip())

        if not valid_texts:
            return outputs

        try:
            self.load_model()
        except Exception as e:
            logger.error("Model unavailable for batch inference: %s", e)
            for idx in valid_indices:
                outputs[idx]["sentiment_status"] = f"model_load_error: {str(e)}"
            return outputs

        # Process in chunks of batch_size
        batch_size = max(1, self.config.batch_size)
        for i in range(0, len(valid_texts), batch_size):
            chunk_texts = valid_texts[i : i + batch_size]
            chunk_indices = valid_indices[i : i + batch_size]

            try:
                chunk_results = self._pipeline(chunk_texts)
                for out_idx, res in zip(chunk_indices, chunk_results):
                    label = str(res.get("label", "unavailable")).strip().lower()
                    score = round(float(res.get("score", 0.0)), 4)
                    outputs[out_idx] = {
                        "sentiment_label": label,
                        "sentiment_score": score,
                        "sentiment_model": self.config.model_name,
                        "sentiment_status": "success",
                    }
            except Exception as e:
                logger.warning("Batch chunk inference failed (%d texts): %s", len(chunk_texts), e)
                # Fallback: process individually to isolate any offending sample
                for out_idx, single_text in zip(chunk_indices, chunk_texts):
                    outputs[out_idx] = self.analyze_text(single_text)

        return outputs

    def enrich_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single commodity news record with sentiment predictions.
        Preserves all original fields untouched.
        """
        if not isinstance(record, dict):
            return {}

        text_to_analyze = record.get("title") or record.get("text")
        sentiment_data = self.analyze_text(text_to_analyze)

        enriched = dict(record)
        enriched.update(sentiment_data)
        return enriched

    def enrich_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich a list of commodity news records using batch inference.
        Preserves all original fields untouched.
        """
        if not records:
            return []

        texts = [r.get("title") or r.get("text") if isinstance(r, dict) else None for r in records]
        sentiments = self.analyze_batch(texts)

        enriched_records = []
        for record, sentiment_data in zip(records, sentiments):
            if not isinstance(record, dict):
                continue
            enriched = dict(record)
            enriched.update(sentiment_data)
            enriched_records.append(enriched)

        return enriched_records
