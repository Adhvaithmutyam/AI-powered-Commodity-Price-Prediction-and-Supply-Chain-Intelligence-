"""
Unit and integration tests for FinBERT Sentiment Analysis using mocked inference.
Ensures zero model downloading or internet access during test runs.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.sentiment.analyzer import FinBERTSentimentAnalyzer, ModelLoadError
from src.sentiment.config import SentimentConfig
from src.sentiment.pipeline import SentimentPipeline


@pytest.fixture
def mock_pipeline():
    """Mock Hugging Face pipeline instance returning predetermined predictions."""
    pipeline = MagicMock()

    def side_effect(inputs, **kwargs):
        if isinstance(inputs, list):
            results = []
            for text in inputs:
                lower = text.lower() if text else ""
                if "surge" in lower or "gain" in lower or "record" in lower:
                    results.append({"label": "positive", "score": 0.9543})
                elif "fall" in lower or "drop" in lower or "decline" in lower:
                    results.append({"label": "negative", "score": 0.8872})
                else:
                    results.append({"label": "neutral", "score": 0.7619})
            return results
        else:
            lower = inputs.lower() if inputs else ""
            if "surge" in lower or "gain" in lower or "record" in lower:
                return [{"label": "positive", "score": 0.9543}]
            elif "fall" in lower or "drop" in lower or "decline" in lower:
                return [{"label": "negative", "score": 0.8872}]
            else:
                return [{"label": "neutral", "score": 0.7619}]

    pipeline.side_effect = side_effect
    return pipeline


def test_analyze_positive_sentiment(mock_pipeline):
    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline)
    result = analyzer.analyze_text("Copper surges to new all-time high amid tight supply")

    assert result["sentiment_label"] == "positive"
    assert result["sentiment_score"] == 0.9543
    assert result["sentiment_model"] == "ProsusAI/finbert"
    assert result["sentiment_status"] == "success"


def test_analyze_negative_sentiment(mock_pipeline):
    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline)
    result = analyzer.analyze_text("Crude oil prices fall sharply on demand worries")

    assert result["sentiment_label"] == "negative"
    assert result["sentiment_score"] == 0.8872
    assert result["sentiment_model"] == "ProsusAI/finbert"
    assert result["sentiment_status"] == "success"


def test_analyze_neutral_sentiment(mock_pipeline):
    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline)
    result = analyzer.analyze_text("Gold markets hold steady ahead of central bank meeting")

    assert result["sentiment_label"] == "neutral"
    assert result["sentiment_score"] == 0.7619
    assert result["sentiment_model"] == "ProsusAI/finbert"
    assert result["sentiment_status"] == "success"


def test_analyze_empty_or_missing_text_does_not_call_model():
    mock_pipeline_unused = MagicMock()
    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline_unused)

    for empty_input in [None, "", "   ", "\n\t"]:
        res = analyzer.analyze_text(empty_input)
        assert res["sentiment_label"] == "unavailable"
        assert res["sentiment_score"] is None
        assert res["sentiment_status"] == "empty_text"

    # Verify model was never invoked
    mock_pipeline_unused.assert_not_called()


def test_analyze_text_model_inference_error():
    mock_pipe = MagicMock()
    mock_pipe.side_effect = RuntimeError("GPU out of memory")

    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipe)
    res = analyzer.analyze_text("Valid headline")

    assert res["sentiment_label"] == "unavailable"
    assert res["sentiment_score"] is None
    assert "error: GPU out of memory" in res["sentiment_status"]


def test_analyze_batch_with_mixed_and_empty_inputs(mock_pipeline):
    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline)
    texts = [
        "Copper price surges",
        "",
        "Oil prices drop",
        None,
        "   ",
        "Gold remains flat",
    ]

    results = analyzer.analyze_batch(texts)
    assert len(results) == 6

    assert results[0]["sentiment_label"] == "positive"
    assert results[1]["sentiment_label"] == "unavailable"
    assert results[1]["sentiment_status"] == "empty_text"
    assert results[2]["sentiment_label"] == "negative"
    assert results[3]["sentiment_label"] == "unavailable"
    assert results[4]["sentiment_label"] == "unavailable"
    assert results[5]["sentiment_label"] == "neutral"


def test_enrich_record_preserves_all_original_fields(mock_pipeline):
    analyzer = FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline)
    raw_record = {
        "commodity": "copper",
        "title": "Copper prices surge on mine disruption",
        "url": "https://reuters.com/markets/copper-surge",
        "publication_timestamp": "2026-10-20T12:00:00Z",
        "domain": "reuters.com",
        "language": "English",
        "source_country": "United States",
        "fetch_timestamp": "2026-10-20T13:00:00Z",
        "provider": "gdelt",
    }

    enriched = analyzer.enrich_record(raw_record)

    # Check that all original keys are perfectly preserved
    for key, val in raw_record.items():
        assert enriched[key] == val

    # Check that new sentiment fields were appended
    assert enriched["sentiment_label"] == "positive"
    assert enriched["sentiment_score"] == 0.9543
    assert enriched["sentiment_model"] == "ProsusAI/finbert"
    assert enriched["sentiment_status"] == "success"


def test_model_loading_error_handling():
    with patch.object(
        FinBERTSentimentAnalyzer,
        "_create_pipeline",
        side_effect=Exception("Model weights not found"),
    ):
        analyzer = FinBERTSentimentAnalyzer()
        with pytest.raises(ModelLoadError) as exc_info:
            analyzer.load_model()
        assert "Could not initialize model" in str(exc_info.value)


def test_sentiment_pipeline_e2e_mocked(tmp_path: Path, mock_pipeline):
    raw_file = tmp_path / "raw" / "commodity_news.jsonl"
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "commodity": "copper",
            "title": "Copper demand surges with EV manufacturing",
            "url": "https://example.com/c1",
            "publication_timestamp": "2026-10-20T10:00:00Z",
        },
        {
            "commodity": "crude oil",
            "title": "Crude oil stockpiles cause price drop",
            "url": "https://example.com/o1",
            "publication_timestamp": "2026-10-20T11:00:00Z",
        },
        {
            "commodity": "gold",
            "title": "",  # Empty title
            "url": "https://example.com/g1",
            "publication_timestamp": "2026-10-20T12:00:00Z",
        },
    ]

    with raw_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    initial_raw_mtime = raw_file.stat().st_mtime
    initial_raw_content = raw_file.read_text(encoding="utf-8")

    processed_file = tmp_path / "processed" / "commodity_news_sentiment.jsonl"
    config = SentimentConfig(
        input_path=raw_file,
        output_path=processed_file,
    )
    analyzer = FinBERTSentimentAnalyzer(config=config, pipeline_instance=mock_pipeline)
    pipeline = SentimentPipeline(config=config, analyzer=analyzer)

    summary = pipeline.run(input_path=raw_file, output_path=processed_file)

    # 1. Summary checks
    assert summary["success"] is True
    assert summary["total_read"] == 3
    assert summary["total_enriched"] == 3
    assert summary["sentiment_counts"]["positive"] == 1
    assert summary["sentiment_counts"]["negative"] == 1
    assert summary["sentiment_counts"]["unavailable"] == 1

    # 2. Output file checks
    assert processed_file.exists()
    with processed_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 3

    p0 = json.loads(lines[0])
    assert p0["commodity"] == "copper"
    assert p0["sentiment_label"] == "positive"
    assert p0["sentiment_score"] == 0.9543

    p1 = json.loads(lines[1])
    assert p1["commodity"] == "crude oil"
    assert p1["sentiment_label"] == "negative"
    assert p1["sentiment_score"] == 0.8872

    p2 = json.loads(lines[2])
    assert p2["commodity"] == "gold"
    assert p2["sentiment_label"] == "unavailable"
    assert p2["sentiment_score"] is None
    assert p2["sentiment_status"] == "empty_text"

    # 3. Verify original raw file was NEVER overwritten or modified
    assert raw_file.read_text(encoding="utf-8") == initial_raw_content


def test_sentiment_pipeline_empty_or_missing_input(tmp_path: Path, mock_pipeline):
    missing_file = tmp_path / "non_existent.jsonl"
    out_file = tmp_path / "out.jsonl"

    pipeline = SentimentPipeline(
        analyzer=FinBERTSentimentAnalyzer(pipeline_instance=mock_pipeline)
    )
    summary = pipeline.run(input_path=missing_file, output_path=out_file)

    assert summary["total_read"] == 0
    assert summary["total_enriched"] == 0
    assert not out_file.exists()
