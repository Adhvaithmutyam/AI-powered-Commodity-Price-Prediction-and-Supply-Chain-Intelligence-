"""
Unit and integration tests for NLP Feature Builder and Timestamp-Safe Aggregator.
All tests run 100% offline with synthetic fixtures and zero network access.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from src.features.builder import ArticleFeatureBuilder, canonicalize_url, load_jsonl_records
from src.features.aggregator import (
    TimestampFeatureAggregator,
    parse_timestamp_utc,
    format_iso_utc,
)
from src.features.pipeline import NLPFeaturePipeline
from src.features.config import FeatureConfig


# ---------------------------------------------------------------------------
# Section 1 & 2 Tests: Article-Level Feature Builder & Joining
# ---------------------------------------------------------------------------

def test_canonicalize_url():
    """Verify URL canonicalization handles trailing slashes, whitespace, and None."""
    assert canonicalize_url("  https://reuters.com/news/copper-strike/  ") == "https://reuters.com/news/copper-strike"
    assert canonicalize_url("https://bloomberg.com/oil") == "https://bloomberg.com/oil"
    assert canonicalize_url(None) == ""
    assert canonicalize_url("") == ""
    # Root domain trailing slash preserved
    assert canonicalize_url("https://reuters.com/") == "https://reuters.com/"


def test_article_feature_builder_complete_join():
    """Verify builder cleanly joins raw, sentiment, and event records on (commodity, url)."""
    raw_news = [
        {
            "commodity": "copper",
            "title": "Chilean copper mine strike halts operations",
            "url": "https://reuters.com/copper-strike",
            "publication_timestamp": "2026-10-20T10:00:00Z",
            "domain": "reuters.com",
            "language": "English",
            "source_country": "Chile",
            "fetch_timestamp": "2026-10-20T11:00:00Z",
            "provider": "gdelt",
        }
    ]
    sentiment_records = [
        {
            "commodity": "copper",
            "title": "Chilean copper mine strike halts operations",
            "url": "https://reuters.com/copper-strike",
            "publication_timestamp": "2026-10-20T10:00:00Z",
            "domain": "reuters.com",
            "language": "English",
            "source_country": "Chile",
            "fetch_timestamp": "2026-10-20T11:00:00Z",
            "provider": "gdelt",
            "sentiment_label": "negative",
            "sentiment_score": 0.9412,
            "sentiment_model": "ProsusAI/finbert",
            "sentiment_status": "success",
        }
    ]
    event_records = [
        {
            "commodity": "copper",
            "event_type": "supply_disruption",
            "event_summary": "Chilean copper mine strike halts operations",
            "impact_direction": "potentially_bullish",
            "confidence": 0.88,
            "evidence_url": "https://reuters.com/copper-strike",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
            "extraction_status": "success",
            "evidence_fact": "mine strike halts operations",
            "implication_rationale": "Stoppage curtails refined copper supply.",
        }
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(
        news_records=raw_news,
        sentiment_records=sentiment_records,
        event_records=event_records,
    )

    assert len(features) == 1
    f = features[0]

    # Verify article identity & traceability
    assert f["commodity"] == "copper"
    assert f["article_url"] == "https://reuters.com/copper-strike"
    assert f["article_title"] == "Chilean copper mine strike halts operations"
    assert f["article_publication_timestamp"] == "2026-10-20T10:00:00Z"
    assert f["domain"] == "reuters.com"
    assert f["language"] == "English"

    # Verify sentiment fields
    assert f["sentiment_label"] == "negative"
    assert f["sentiment_score"] == 0.9412
    assert f["sentiment_status"] == "success"

    # Verify relevance fields
    assert f["relevance_class"] == "directly_relevant"
    assert f["relevance_score"] >= 0.85
    assert "mining_production" in f["matched_relevance_indicators"]

    # Verify event fields
    assert f["event_type"] == "supply_disruption"
    assert f["event_impact_direction"] == "potentially_bullish"
    assert f["event_confidence"] == 0.88
    assert f["evidence_fact"] == "mine strike halts operations"
    assert f["feature_status"] == "complete"


def test_article_feature_builder_missing_sentiment():
    """Verify builder handles missing sentiment data with explicit missing status."""
    raw_news = [
        {
            "commodity": "gold",
            "title": "Central banks add to gold bullion reserves",
            "url": "https://ft.com/gold-reserves",
            "publication_timestamp": "2026-10-20T08:00:00Z",
        }
    ]
    event_records = [
        {
            "commodity": "gold",
            "event_type": "demand_change",
            "evidence_url": "https://ft.com/gold-reserves",
            "impact_direction": "potentially_bullish",
            "confidence": 0.85,
        }
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(news_records=raw_news, event_records=event_records)

    assert len(features) == 1
    f = features[0]
    assert f["commodity"] == "gold"
    assert f["sentiment_label"] is None
    assert f["sentiment_score"] is None
    assert f["sentiment_status"] == "missing"
    assert f["event_type"] == "demand_change"
    assert f["feature_status"] == "partial_missing_sentiment"


def test_article_feature_builder_missing_events():
    """Verify builder handles missing event data with fallback defaults."""
    sentiment_records = [
        {
            "commodity": "crude oil",
            "title": "Oil markets volatile amid Middle East tensions",
            "url": "https://bloomberg.com/oil-tensions",
            "publication_timestamp": "2026-10-20T09:30:00Z",
            "sentiment_label": "negative",
            "sentiment_score": 0.82,
            "sentiment_status": "success",
        }
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(sentiment_records=sentiment_records)

    assert len(features) == 1
    f = features[0]
    assert f["commodity"] == "crude oil"
    assert f["sentiment_label"] == "negative"
    assert f["event_type"] == "other"
    assert f["event_impact_direction"] == "unknown"
    assert f["event_confidence"] == 0.0
    assert f["extraction_status"] == "missing"
    assert f["feature_status"] == "partial_missing_events"


def test_article_feature_builder_missing_both():
    """Verify builder handles raw-only news articles missing both sentiment and events."""
    raw_news = [
        {
            "commodity": "copper",
            "title": "Copper inventories rise on Asian exchanges",
            "url": "https://metalbulletin.com/copper-inventories",
            "publication_timestamp": "2026-10-20T07:15:00Z",
        }
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(news_records=raw_news)

    assert len(features) == 1
    f = features[0]
    assert f["sentiment_status"] == "missing"
    assert f["extraction_status"] == "missing"
    assert f["feature_status"] == "partial_missing_both"
    # Dynamic relevance should still be evaluated from title
    assert f["relevance_class"] == "directly_relevant"
    assert "inventory_storage" in f["matched_relevance_indicators"]


def test_article_feature_builder_event_only():
    """Verify builder gracefully incorporates articles that only exist in the event dataset."""
    event_records = [
        {
            "commodity": "crude oil",
            "event_type": "supply_disruption",
            "event_summary": "Pipeline explosion stops crude oil transit",
            "evidence_url": "https://oilprice.com/pipeline-explosion",
            "article_publication_timestamp": "2026-10-20T11:45:00Z",
            "impact_direction": "potentially_bullish",
            "confidence": 0.92,
        }
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(event_records=event_records)

    assert len(features) == 1
    f = features[0]
    assert f["commodity"] == "crude oil"
    assert f["article_url"] == "https://oilprice.com/pipeline-explosion"
    assert f["article_title"] == "Pipeline explosion stops crude oil transit"
    assert f["event_type"] == "supply_disruption"
    assert f["event_impact_direction"] == "potentially_bullish"
    assert f["feature_status"] == "partial_missing_sentiment"


def test_article_feature_builder_deduplication():
    """Verify articles with duplicate (commodity, url) keys are deduplicated."""
    raw_news = [
        {"commodity": "gold", "title": "Gold news 1", "url": "https://news.com/gold-1"},
        {"commodity": "gold", "title": "Gold news 1 dup", "url": "https://news.com/gold-1"},
        {"commodity": "copper", "title": "Copper news 1", "url": "https://news.com/gold-1"},  # distinct commodity allowed
    ]
    builder = ArticleFeatureBuilder()
    features = builder.build(news_records=raw_news)

    # 1 gold, 1 copper
    assert len(features) == 2
    commodities = [f["commodity"] for f in features]
    assert commodities == ["gold", "copper"]


# ---------------------------------------------------------------------------
# Section 3 & 4 Tests: Timestamp Safety, Aggregation & Leakage Prevention
# ---------------------------------------------------------------------------

def test_parse_timestamp_utc_various_formats():
    """Verify timestamp parsing handles standard, offset, naive, date-only, and invalid values."""
    # Standard ISO UTC
    dt1 = parse_timestamp_utc("2026-10-20T12:45:00Z")
    assert dt1 == datetime(2026, 10, 20, 12, 45, 0, tzinfo=timezone.utc)

    # ISO with positive offset (+02:00) converted to UTC
    dt2 = parse_timestamp_utc("2026-10-20T14:45:00+02:00")
    assert dt2 == datetime(2026, 10, 20, 12, 45, 0, tzinfo=timezone.utc)

    # Timezone-naive datetime string
    dt3 = parse_timestamp_utc("2026-10-20 12:45:00")
    assert dt3 == datetime(2026, 10, 20, 12, 45, 0, tzinfo=timezone.utc)

    # Date only string
    dt4 = parse_timestamp_utc("2026-10-20")
    assert dt4 == datetime(2026, 10, 20, 0, 0, 0, tzinfo=timezone.utc)

    # Datetime instance without tzinfo
    dt5 = parse_timestamp_utc(datetime(2026, 10, 20, 12, 45, 0))
    assert dt5 == datetime(2026, 10, 20, 12, 45, 0, tzinfo=timezone.utc)

    # Invalid, None, or empty
    assert parse_timestamp_utc(None) is None
    assert parse_timestamp_utc("") is None
    assert parse_timestamp_utc("not-a-timestamp") is None
    assert parse_timestamp_utc(12345) is None


def test_timestamp_aggregation_exact_window_boundaries():
    """
    Verify articles published precisely at window_end (cutoff) and window_start are included,
    while articles 1 second outside are excluded.
    """
    cutoff = "2026-10-20T12:00:00Z"
    # 1 day window: 2026-10-19T12:00:00Z to 2026-10-20T12:00:00Z

    articles = [
        # Exactly at cutoff (window_end) -> INCLUDED
        {"commodity": "copper", "article_url": "https://a.com/1", "article_publication_timestamp": "2026-10-20T12:00:00Z"},
        # Mid-window -> INCLUDED
        {"commodity": "copper", "article_url": "https://a.com/2", "article_publication_timestamp": "2026-10-20T00:00:00Z"},
        # Exactly at window_start -> INCLUDED
        {"commodity": "copper", "article_url": "https://a.com/3", "article_publication_timestamp": "2026-10-19T12:00:00Z"},
        # 1 second before window_start -> EXCLUDED (too old)
        {"commodity": "copper", "article_url": "https://a.com/4", "article_publication_timestamp": "2026-10-19T11:59:59Z"},
        # 1 second after cutoff -> EXCLUDED (future leakage)
        {"commodity": "copper", "article_url": "https://a.com/5", "article_publication_timestamp": "2026-10-20T12:00:01Z"},
    ]

    aggregator = TimestampFeatureAggregator()
    res = aggregator.aggregate(
        article_features=articles,
        commodity="copper",
        cutoff_timestamp=cutoff,
        window_days=1,
    )

    assert res["article_count"] == 3
    assert res["window_start_timestamp"] == "2026-10-19T12:00:00Z"
    assert res["window_end_timestamp"] == "2026-10-20T12:00:00Z"
    assert res["future_articles_filtered"] == 1


def test_timestamp_aggregation_future_leakage_exclusion():
    """Verify articles published in the future relative to the cutoff are strictly excluded."""
    cutoff = "2026-10-15T00:00:00Z"

    articles = [
        {"commodity": "gold", "article_url": "https://a.com/past", "article_publication_timestamp": "2026-10-14T20:00:00Z"},
        {"commodity": "gold", "article_url": "https://a.com/future-1", "article_publication_timestamp": "2026-10-15T00:01:00Z"},
        {"commodity": "gold", "article_url": "https://a.com/future-2", "article_publication_timestamp": "2026-10-16T12:00:00Z"},
    ]

    aggregator = TimestampFeatureAggregator()
    res = aggregator.aggregate(
        article_features=articles,
        commodity="gold",
        cutoff_timestamp=cutoff,
        window_days=5,
    )

    # Only the past article should be included
    assert res["article_count"] == 1
    assert res["future_articles_filtered"] == 2


def test_timestamp_aggregation_unanchored_exclusion():
    """Verify articles with missing or invalid publication timestamps are excluded from time windows."""
    cutoff = "2026-10-20T12:00:00Z"

    articles = [
        {"commodity": "copper", "article_url": "https://a.com/valid", "article_publication_timestamp": "2026-10-20T10:00:00Z"},
        {"commodity": "copper", "article_url": "https://a.com/none", "article_publication_timestamp": None},
        {"commodity": "copper", "article_url": "https://a.com/empty", "article_publication_timestamp": ""},
        {"commodity": "copper", "article_url": "https://a.com/bad", "article_publication_timestamp": "invalid-timestamp"},
    ]

    aggregator = TimestampFeatureAggregator()
    res = aggregator.aggregate(
        article_features=articles,
        commodity="copper",
        cutoff_timestamp=cutoff,
        window_days=1,
    )

    assert res["article_count"] == 1
    assert res["unanchored_articles_excluded"] == 3


def test_timestamp_aggregation_no_duplicate_counting():
    """Verify multiple instances of the same URL in a window are deduplicated and counted once."""
    cutoff = "2026-10-20T12:00:00Z"

    articles = [
        {"commodity": "crude oil", "article_url": "https://a.com/oil-1", "article_publication_timestamp": "2026-10-20T08:00:00Z"},
        {"commodity": "crude oil", "article_url": "https://a.com/oil-1", "article_publication_timestamp": "2026-10-20T09:00:00Z"},
        {"commodity": "crude oil", "article_url": "https://a.com/oil-2", "article_publication_timestamp": "2026-10-20T10:00:00Z"},
    ]

    aggregator = TimestampFeatureAggregator()
    res = aggregator.aggregate(
        article_features=articles,
        commodity="crude oil",
        cutoff_timestamp=cutoff,
        window_days=1,
    )

    assert res["article_count"] == 2


def test_timestamp_aggregation_metric_calculations():
    """Verify calculations of relevance, sentiment, event counts, means, and polarity."""
    cutoff = "2026-10-20T12:00:00Z"

    articles = [
        {
            "commodity": "copper",
            "article_url": "https://a.com/1",
            "article_publication_timestamp": "2026-10-20T08:00:00Z",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.90,
            "sentiment_label": "positive",
            "sentiment_score": 0.80,
            "event_type": "supply_disruption",
            "event_impact_direction": "potentially_bullish",
            "event_confidence": 0.85,
        },
        {
            "commodity": "copper",
            "article_url": "https://a.com/2",
            "article_publication_timestamp": "2026-10-20T09:00:00Z",
            "relevance_class": "potentially_relevant",
            "relevance_score": 0.70,
            "sentiment_label": "negative",
            "sentiment_score": 0.90,
            "event_type": "demand_change",
            "event_impact_direction": "potentially_bearish",
            "event_confidence": 0.75,
        },
        {
            "commodity": "copper",
            "article_url": "https://a.com/3",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
            "relevance_class": "irrelevant",
            "relevance_score": 0.10,
            "sentiment_label": "neutral",
            "sentiment_score": 0.70,
            "event_type": "other",
            "event_impact_direction": "unknown",
            "event_confidence": 0.0,
        },
    ]

    aggregator = TimestampFeatureAggregator(confidence_threshold=0.70)
    res = aggregator.aggregate(
        article_features=articles,
        commodity="copper",
        cutoff_timestamp=cutoff,
        window_days=1,
    )

    assert res["article_count"] == 3
    assert res["relevant_article_count"] == 2  # directly_relevant + potentially_relevant
    assert res["directly_relevant_count"] == 1
    assert res["potentially_relevant_count"] == 1
    assert res["irrelevant_count"] == 1
    assert res["uncertain_count"] == 0
    assert res["relevance_score_mean"] == round((0.90 + 0.70 + 0.10) / 3, 4)

    # Sentiment metrics
    assert res["sentiment_positive_count"] == 1
    assert res["sentiment_negative_count"] == 1
    assert res["sentiment_neutral_count"] == 1
    assert res["sentiment_missing_count"] == 0
    assert res["sentiment_confidence_mean"] == round((0.80 + 0.90 + 0.70) / 3, 4)
    assert res["sentiment_score_mean"] == round((0.80 + 0.90 + 0.70) / 3, 4)
    # Net polarity: (1 - 1) / 3 = 0.0
    assert res["net_sentiment_polarity"] == 0.0
    # Confidence-weighted polarity: (+0.80 - 0.90 + 0.0) / 3 = -0.10 / 3 = -0.0333
    assert res["confidence_weighted_sentiment_polarity"] == round(-0.10 / 3, 4)

    # Relevant-only sentiment metrics
    assert res["relevant_sentiment_positive_count"] == 1
    assert res["relevant_sentiment_negative_count"] == 1
    assert res["relevant_sentiment_neutral_count"] == 0
    assert res["relevant_net_sentiment_polarity"] == 0.0
    # Relevant confidence-weighted polarity: (+0.80 - 0.90) / 2 = -0.05
    assert res["relevant_confidence_weighted_polarity"] == -0.05

    # Event metrics
    assert res["event_type_counts"]["supply_disruption"] == 1
    assert res["event_type_counts"]["demand_change"] == 1
    assert res["event_type_counts"]["other"] == 1
    assert res["event_impact_counts"]["potentially_bullish"] == 1
    assert res["event_impact_counts"]["potentially_bearish"] == 1
    assert res["event_impact_counts"]["unknown"] == 1
    assert res["high_confidence_event_count"] == 2  # articles 1 and 2 (conf >= 0.70 and not 'other')


def test_timestamp_aggregation_relevant_sentiment_isolation():
    """
    Verify that an irrelevant article (e.g., sports gold medal victory with positive sentiment)
    does not pollute relevant-only sentiment and polarity signals.
    """
    cutoff = "2026-10-20T12:00:00Z"
    articles = [
        {
            "commodity": "gold",
            "article_url": "https://mining.com/gold-mine-disruption",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.95,
            "sentiment_label": "negative",
            "sentiment_score": 0.90,
            "event_type": "supply_disruption",
            "event_impact_direction": "potentially_bullish",
            "event_confidence": 0.88,
        },
        {
            "commodity": "gold",
            "article_url": "https://sports.com/olympic-gold-medal",
            "article_publication_timestamp": "2026-10-20T11:00:00Z",
            "relevance_class": "irrelevant",
            "relevance_score": 0.05,
            "sentiment_label": "positive",
            "sentiment_score": 0.95,
            "event_type": "other",
            "event_impact_direction": "unknown",
            "event_confidence": 0.0,
        },
    ]

    aggregator = TimestampFeatureAggregator()
    res = aggregator.aggregate(
        article_features=articles,
        commodity="gold",
        cutoff_timestamp=cutoff,
        window_days=1,
    )

    # All articles
    assert res["article_count"] == 2
    assert res["sentiment_positive_count"] == 1
    assert res["sentiment_negative_count"] == 1
    assert res["net_sentiment_polarity"] == 0.0  # (1 - 1) / 2 = 0.0

    # Relevant only (sports article isolated)
    assert res["relevant_article_count"] == 1
    assert res["relevant_sentiment_positive_count"] == 0
    assert res["relevant_sentiment_negative_count"] == 1
    assert res["relevant_net_sentiment_polarity"] == -1.0  # (0 - 1) / 1 = -1.0
    assert res["relevant_confidence_weighted_polarity"] == -0.90


def test_timestamp_aggregation_cutoff_source():
    """Verify cutoff_source accurately reflects explicit cutoff vs latest article derivation."""
    articles = [
        {
            "commodity": "copper",
            "article_url": "https://news.com/c1",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
        }
    ]
    aggregator = TimestampFeatureAggregator()

    # Explicit cutoff
    explicit_res = aggregator.aggregate(
        article_features=articles,
        commodity="copper",
        cutoff_timestamp="2026-10-20T12:00:00Z",
        window_days=1,
    )
    assert explicit_res["cutoff_source"] == "explicit"

    # Default latest article derivation via aggregate_windows
    windows = aggregator.aggregate_windows(
        article_features=articles,
        commodities=["copper"],
        cutoff_timestamp=None,
        window_days_list=[1],
    )
    assert len(windows) == 1
    assert windows[0]["cutoff_source"] == "default_latest_article"
    assert windows[0]["cutoff_timestamp"] == "2026-10-20T10:00:00Z"


def test_timestamp_aggregation_empty_window():
    """Verify aggregator returns zeros and None means when no articles fall in window."""
    cutoff = "2026-10-20T12:00:00Z"
    aggregator = TimestampFeatureAggregator()
    res = aggregator.aggregate(
        article_features=[],
        commodity="gold",
        cutoff_timestamp=cutoff,
        window_days=1,
    )

    assert res["article_count"] == 0
    assert res["relevant_article_count"] == 0
    assert res["relevance_score_mean"] is None
    assert res["sentiment_score_mean"] is None
    assert res["net_sentiment_polarity"] == 0.0
    assert res["high_confidence_event_count"] == 0


def test_aggregate_windows_multi_commodity_and_horizons():
    """Verify aggregate_windows produces cross-product of commodities and window horizons."""
    articles = [
        {"commodity": "copper", "article_url": "https://a.com/c1", "article_publication_timestamp": "2026-10-20T10:00:00Z"},
        {"commodity": "gold", "article_url": "https://a.com/g1", "article_publication_timestamp": "2026-10-20T10:00:00Z"},
    ]

    aggregator = TimestampFeatureAggregator()
    res_list = aggregator.aggregate_windows(
        article_features=articles,
        commodities=["copper", "gold", "crude oil"],
        cutoff_timestamp="2026-10-20T12:00:00Z",
        window_days_list=[1, 5],
    )

    # 3 commodities * 2 windows = 6 records
    assert len(res_list) == 6
    copper_1d = [r for r in res_list if r["commodity"] == "copper" and r["window_days"] == 1][0]
    assert copper_1d["article_count"] == 1
    oil_5d = [r for r in res_list if r["commodity"] == "crude oil" and r["window_days"] == 5][0]
    assert oil_5d["article_count"] == 0


# ---------------------------------------------------------------------------
# Section 5 Tests: File Loading, Pipeline Integration & Non-Destructive Flow
# ---------------------------------------------------------------------------

def test_load_jsonl_records_malformed_and_missing(tmp_path: Path):
    """Verify load_jsonl_records ignores empty lines, skips malformed JSON, and handles missing files."""
    missing_file = tmp_path / "non_existent.jsonl"
    assert load_jsonl_records(missing_file) == []

    test_file = tmp_path / "corrupt.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write('{"commodity": "copper", "title": "Good"}\n')
        f.write('\n')  # Empty line
        f.write('INVALID JSON LINE\n')  # Malformed line
        f.write('{"commodity": "gold", "title": "Also Good"}\n')

    loaded = load_jsonl_records(test_file)
    assert len(loaded) == 2
    assert loaded[0]["commodity"] == "copper"
    assert loaded[1]["commodity"] == "gold"


def test_nlp_feature_pipeline_e2e(tmp_path: Path):
    """Verify NLPFeaturePipeline end-to-end execution writing both article and aggregated datasets."""
    raw_file = tmp_path / "raw.jsonl"
    sent_file = tmp_path / "sentiment.jsonl"
    event_file = tmp_path / "events.jsonl"
    out_art = tmp_path / "nlp_article_features.jsonl"
    out_agg = tmp_path / "nlp_aggregated_features.jsonl"

    record = {
        "commodity": "copper",
        "title": "Strike halts Escondida copper mine operations",
        "url": "https://news.com/escondida-strike",
        "publication_timestamp": "2026-10-20T10:00:00Z",
        "domain": "news.com",
        "language": "English",
        "source_country": "Chile",
        "fetch_timestamp": "2026-10-20T11:00:00Z",
        "provider": "gdelt",
        "sentiment_label": "negative",
        "sentiment_score": 0.91,
        "sentiment_model": "ProsusAI/finbert",
        "sentiment_status": "success",
    }
    with open(sent_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    event_record = {
        "commodity": "copper",
        "event_type": "supply_disruption",
        "event_summary": "Strike halts Escondida copper mine operations",
        "impact_direction": "potentially_bullish",
        "confidence": 0.88,
        "evidence_url": "https://news.com/escondida-strike",
        "article_publication_timestamp": "2026-10-20T10:00:00Z",
        "extraction_status": "success",
    }
    with open(event_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(event_record) + "\n")

    pipeline = NLPFeaturePipeline()
    summary = pipeline.run(
        raw_news_path=raw_file,
        sentiment_path=sent_file,
        events_path=event_file,
        output_article_path=out_art,
        output_aggregated_path=out_agg,
        cutoff_timestamp="2026-10-20T12:00:00Z",
        window_days_list=[1, 5],
        commodities=["copper"],
        build_aggregated=True,
    )

    assert summary["success"] is True
    assert summary["article_features_count"] == 1
    assert summary["aggregated_features_count"] == 2
    assert out_art.exists()
    assert out_agg.exists()

    # Read back generated files
    with open(out_art, "r", encoding="utf-8") as f:
        art_rows = [json.loads(line) for line in f if line.strip()]
    assert len(art_rows) == 1
    assert art_rows[0]["relevance_class"] == "directly_relevant"
    assert art_rows[0]["event_type"] == "supply_disruption"
    assert art_rows[0]["feature_status"] == "complete"

    with open(out_agg, "r", encoding="utf-8") as f:
        agg_rows = [json.loads(line) for line in f if line.strip()]
    assert len(agg_rows) == 2
    assert agg_rows[0]["article_count"] == 1
    assert agg_rows[0]["event_impact_counts"]["potentially_bullish"] == 1


# ---------------------------------------------------------------------------
# Section 6 Tests: Regression Tests for Issue 1 (Merging) & Issue 2 (Null Safety)
# ---------------------------------------------------------------------------

def test_article_feature_builder_partial_sentiment_merge_preserves_all_raw_articles():
    """
    Regression Test (Issue 1):
    When 5 raw articles and 3 matching sentiment articles are provided,
    all 5 unique raw articles must be preserved in the feature dataset.
    Overlapping articles must not be duplicated, and raw-only articles must
    correctly report missing sentiment status.
    """
    raw_news = [
        {
            "commodity": "copper",
            "title": f"Copper market article {i}",
            "url": f"https://mining.com/copper-{i}",
            "publication_timestamp": f"2026-10-20T0{i}:00:00Z",
            "domain": "mining.com",
            "language": "English",
            "source_country": "Chile",
            "fetch_timestamp": "2026-10-20T11:00:00Z",
            "provider": "gdelt",
        }
        for i in range(1, 6)
    ]

    sentiment_records = [
        {
            "commodity": "copper",
            "title": f"Copper market article {i}",
            "url": f"https://mining.com/copper-{i}",
            "publication_timestamp": f"2026-10-20T0{i}:00:00Z",
            "domain": "mining.com",
            "language": "English",
            "source_country": "Chile",
            "fetch_timestamp": "2026-10-20T11:00:00Z",
            "provider": "gdelt",
            "sentiment_label": "positive" if i % 2 == 1 else "negative",
            "sentiment_score": 0.85 + (i * 0.02),
            "sentiment_model": "ProsusAI/finbert",
            "sentiment_status": "success",
        }
        for i in range(1, 4)  # articles 1, 2, 3 have sentiment; 4, 5 are raw-only
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(news_records=raw_news, sentiment_records=sentiment_records)

    # 1. All 5 articles must remain
    assert len(features) == 5

    # 2. Overlapping articles must not be duplicated
    urls = [f["article_url"] for f in features]
    assert len(set(urls)) == 5
    assert urls == [f"https://mining.com/copper-{i}" for i in range(1, 6)]

    # 3. Matching articles (1, 2, 3) must have sentiment-enriched information
    for f in features[:3]:
        assert f["sentiment_status"] == "success"
        assert f["sentiment_label"] in ("positive", "negative")
        assert f["sentiment_score"] is not None
        assert f["sentiment_model"] == "ProsusAI/finbert"
        assert f["feature_status"] == "partial_missing_events"

    # 4. Raw-only articles (4, 5) must indicate missing sentiment
    for f in features[3:]:
        assert f["sentiment_status"] == "missing"
        assert f["sentiment_label"] is None
        assert f["sentiment_score"] is None
        assert f["sentiment_model"] is None
        assert f["feature_status"] == "partial_missing_both"
        # Metadata from raw news preserved
        assert f["domain"] == "mining.com"
        assert f["source_country"] == "Chile"


def test_article_feature_builder_sentiment_only_and_event_only_coexistence():
    """
    Regression Test (Issue 1):
    Ensure that sentiment-only records and event-only records remain fully supported
    even alongside raw news records.
    """
    raw_news = [
        {
            "commodity": "gold",
            "title": "Gold reserves headline",
            "url": "https://news.com/gold-raw",
            "publication_timestamp": "2026-10-20T08:00:00Z",
        }
    ]
    sentiment_records = [
        {
            "commodity": "gold",
            "title": "Gold sentiment headline",
            "url": "https://news.com/gold-sentiment-only",
            "publication_timestamp": "2026-10-20T09:00:00Z",
            "sentiment_label": "positive",
            "sentiment_score": 0.92,
            "sentiment_status": "success",
        }
    ]
    event_records = [
        {
            "commodity": "gold",
            "event_type": "supply_disruption",
            "event_summary": "Mine shutdown event headline",
            "impact_direction": "potentially_bullish",
            "confidence": 0.89,
            "evidence_url": "https://news.com/gold-event-only",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
        }
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(
        news_records=raw_news,
        sentiment_records=sentiment_records,
        event_records=event_records,
    )

    assert len(features) == 3
    urls = [f["article_url"] for f in features]
    assert "https://news.com/gold-raw" in urls
    assert "https://news.com/gold-sentiment-only" in urls
    assert "https://news.com/gold-event-only" in urls

    raw_feat = next(f for f in features if f["article_url"] == "https://news.com/gold-raw")
    assert raw_feat["sentiment_status"] == "missing"
    assert raw_feat["feature_status"] == "partial_missing_both"

    sent_feat = next(f for f in features if f["article_url"] == "https://news.com/gold-sentiment-only")
    assert sent_feat["sentiment_status"] == "success"
    assert sent_feat["feature_status"] == "partial_missing_events"

    ev_feat = next(f for f in features if f["article_url"] == "https://news.com/gold-event-only")
    assert ev_feat["sentiment_status"] == "missing"
    assert ev_feat["event_type"] == "supply_disruption"
    assert ev_feat["feature_status"] == "partial_missing_sentiment"


def test_article_feature_builder_multi_commodity_identities_preserved():
    """
    Regression Test (Issue 1):
    When a single macro headline references multiple commodities, each commodity
    must retain its distinct commodity identity, relevance, and event/sentiment mapping.
    """
    shared_url = "https://reuters.com/macro/fed-rate-decision"
    raw_news = [
        {
            "commodity": "copper",
            "title": "Fed cuts rates boosting metals and energy demand",
            "url": shared_url,
            "publication_timestamp": "2026-10-20T14:00:00Z",
        },
        {
            "commodity": "gold",
            "title": "Fed cuts rates boosting metals and energy demand",
            "url": shared_url,
            "publication_timestamp": "2026-10-20T14:00:00Z",
        },
        {
            "commodity": "crude oil",
            "title": "Fed cuts rates boosting metals and energy demand",
            "url": shared_url,
            "publication_timestamp": "2026-10-20T14:00:00Z",
        },
    ]
    # Sentiment only available for copper and gold
    sentiment_records = [
        {
            "commodity": "copper",
            "url": shared_url,
            "sentiment_label": "positive",
            "sentiment_score": 0.88,
            "sentiment_status": "success",
        },
        {
            "commodity": "gold",
            "url": shared_url,
            "sentiment_label": "positive",
            "sentiment_score": 0.95,
            "sentiment_status": "success",
        },
    ]

    builder = ArticleFeatureBuilder()
    features = builder.build(news_records=raw_news, sentiment_records=sentiment_records)

    assert len(features) == 3
    commodities = {f["commodity"] for f in features}
    assert commodities == {"copper", "gold", "crude oil"}

    copper_feat = next(f for f in features if f["commodity"] == "copper")
    assert copper_feat["sentiment_status"] == "success"
    assert copper_feat["sentiment_score"] == 0.88

    gold_feat = next(f for f in features if f["commodity"] == "gold")
    assert gold_feat["sentiment_status"] == "success"
    assert gold_feat["sentiment_score"] == 0.95

    oil_feat = next(f for f in features if f["commodity"] == "crude oil")
    assert oil_feat["sentiment_status"] == "missing"
    assert oil_feat["feature_status"] == "partial_missing_both"


def test_article_feature_builder_matched_indicators_null_safe():
    """
    Regression Test (Issue 2):
    Verify that matched_indicators handles None/null, missing, single strings,
    malformed types, and valid lists without raising TypeError.
    """
    builder = ArticleFeatureBuilder()

    records = [
        # Explicit None (null in JSON)
        {
            "commodity": "copper",
            "title": "Copper mine strike halts operations",
            "url": "https://news.com/null-indicators",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.90,
            "relevance_explanation": "Test explanation",
            "matched_indicators": None,
        },
        # Missing matched_indicators entirely
        {
            "commodity": "copper",
            "title": "Copper smelter outage",
            "url": "https://news.com/missing-indicators",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.85,
        },
        # Empty list
        {
            "commodity": "copper",
            "title": "Copper price stable",
            "url": "https://news.com/empty-indicators",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.80,
            "matched_indicators": [],
        },
        # Single string instead of list (malformed)
        {
            "commodity": "copper",
            "title": "Copper inventories rise",
            "url": "https://news.com/string-indicator",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.80,
            "matched_indicators": "mining_production",
        },
        # Numeric/malformed type
        {
            "commodity": "copper",
            "title": "Copper demand unchanged",
            "url": "https://news.com/numeric-indicator",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.80,
            "matched_indicators": 12345,
        },
        # Valid list with some None/empty elements
        {
            "commodity": "copper",
            "title": "Copper export ban",
            "url": "https://news.com/valid-list-indicators",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.92,
            "matched_indicators": ["commodity_copper", None, "", "mining_production"],
        },
    ]

    features = builder.build(news_records=records)

    assert len(features) == 6

    # 1. Explicit None -> empty list
    assert features[0]["matched_relevance_indicators"] == []

    # 2. Missing -> empty list
    assert features[1]["matched_relevance_indicators"] == []

    # 3. Empty list -> empty list
    assert features[2]["matched_relevance_indicators"] == []

    # 4. Single string -> single element list
    assert features[3]["matched_relevance_indicators"] == ["mining_production"]

    # 5. Malformed number -> empty list
    assert features[4]["matched_relevance_indicators"] == []

    # 6. Valid list with dirty elements -> sanitized list
    assert features[5]["matched_relevance_indicators"] == ["commodity_copper", "mining_production"]
