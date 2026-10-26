"""
Integration tests for NewsIngestionPipeline with mocked GDELTClient.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.news_intelligence.config import PipelineConfig
from src.news_intelligence.gdelt_client import GDELTClient, GDELTNetworkError
from src.news_intelligence.pipeline import NewsIngestionPipeline
from src.news_intelligence.storage import load_existing_keys


def test_pipeline_e2e_mocked_success(tmp_path: Path):
    output_file = tmp_path / "commodity_news.jsonl"
    config = PipelineConfig(
        commodities=["copper", "gold", "crude oil"],
        output_path=output_file,
        rate_limit_delay_seconds=0.0,
    )

    mock_client = MagicMock(spec=GDELTClient)

    def mock_fetch(query, **kwargs):
        if "copper" in query:
            return [
                {
                    "title": "Copper demand surges",
                    "url": "https://news.com/copper-1",
                    "seendate": "20261015T100000Z",
                    "domain": "news.com",
                    "language": "English",
                    "sourcecountry": "Chile",
                }
            ]
        elif "gold" in query:
            return [
                {
                    "title": "Gold reaches all-time high",
                    "url": "https://news.com/gold-1",
                    "seendate": "20261015T110000Z",
                    "domain": "news.com",
                    "language": "English",
                    "sourcecountry": "United States",
                }
            ]
        elif "crude oil" in query:
            return [
                {
                    "title": "OPEC oil production update",
                    "url": "https://news.com/oil-1",
                    "seendate": "20261015T120000Z",
                    "domain": "news.com",
                    "language": "English",
                    "sourcecountry": "Saudi Arabia",
                }
            ]
        return []

    mock_client.fetch_articles.side_effect = mock_fetch

    pipeline = NewsIngestionPipeline(config=config, client=mock_client)
    summary = pipeline.run(output_path=output_file)

    assert summary["success"] is True
    assert summary["total_fetched"] == 3
    assert summary["total_saved"] == 3
    assert summary["total_duplicates"] == 0
    assert len(summary["errors"]) == 0

    assert summary["commodity_stats"]["copper"]["saved"] == 1
    assert summary["commodity_stats"]["gold"]["saved"] == 1
    assert summary["commodity_stats"]["crude oil"]["saved"] == 1

    # Verify keys saved to storage
    existing_keys = load_existing_keys(output_file)
    assert len(existing_keys) == 3
    assert ("copper", "https://news.com/copper-1") in existing_keys
    assert ("gold", "https://news.com/gold-1") in existing_keys
    assert ("crude oil", "https://news.com/oil-1") in existing_keys


def test_pipeline_partial_failure_continues(tmp_path: Path):
    output_file = tmp_path / "commodity_news.jsonl"
    config = PipelineConfig(
        commodities=["copper", "gold", "crude oil"],
        output_path=output_file,
        rate_limit_delay_seconds=0.0,
    )

    mock_client = MagicMock(spec=GDELTClient)

    def mock_fetch(query, **kwargs):
        if "copper" in query:
            raise GDELTNetworkError("Connection timed out for copper query")
        elif "gold" in query:
            return [
                {
                    "title": "Gold price gains",
                    "url": "https://news.com/gold-news",
                    "seendate": "20261015T090000Z",
                }
            ]
        elif "crude oil" in query:
            return [
                {
                    "title": "Crude oil stockpiles fall",
                    "url": "https://news.com/oil-news",
                    "seendate": "20261015T093000Z",
                }
            ]
        return []

    mock_client.fetch_articles.side_effect = mock_fetch

    pipeline = NewsIngestionPipeline(config=config, client=mock_client)
    summary = pipeline.run(output_path=output_file)

    # Gold and crude oil should still succeed and be saved
    assert summary["total_saved"] == 2
    assert summary["commodity_stats"]["copper"]["fetched"] == 0
    assert "error" in summary["commodity_stats"]["copper"]
    assert "copper" in summary["errors"]
    assert summary["commodity_stats"]["gold"]["saved"] == 1
    assert summary["commodity_stats"]["crude oil"]["saved"] == 1

    existing_keys = load_existing_keys(output_file)
    assert len(existing_keys) == 2
    assert ("gold", "https://news.com/gold-news") in existing_keys
    assert ("crude oil", "https://news.com/oil-news") in existing_keys


def test_pipeline_deduplication_on_repeated_run(tmp_path: Path):
    output_file = tmp_path / "commodity_news.jsonl"
    config = PipelineConfig(
        commodities=["gold"],
        output_path=output_file,
        rate_limit_delay_seconds=0.0,
    )

    mock_client = MagicMock(spec=GDELTClient)
    mock_client.fetch_articles.return_value = [
        {
            "title": "Gold article 1",
            "url": "https://news.com/gold-article",
            "seendate": "20261015T120000Z",
        }
    ]

    pipeline = NewsIngestionPipeline(config=config, client=mock_client)

    # First run: saves article
    summary1 = pipeline.run(output_path=output_file)
    assert summary1["total_saved"] == 1
    assert summary1["total_duplicates"] == 0

    # Second run: same article fetched again -> skipped as duplicate
    summary2 = pipeline.run(output_path=output_file)
    assert summary2["total_saved"] == 0
    assert summary2["total_duplicates"] == 1


def test_pipeline_empty_articles_handled(tmp_path: Path):
    output_file = tmp_path / "commodity_news.jsonl"
    config = PipelineConfig(
        commodities=["copper"],
        output_path=output_file,
        rate_limit_delay_seconds=0.0,
    )

    mock_client = MagicMock(spec=GDELTClient)
    mock_client.fetch_articles.return_value = []

    pipeline = NewsIngestionPipeline(config=config, client=mock_client)
    summary = pipeline.run(output_path=output_file)

    assert summary["total_fetched"] == 0
    assert summary["total_saved"] == 0
    assert summary["total_duplicates"] == 0
    assert summary["success"] is True
