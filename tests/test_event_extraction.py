"""
Unit and integration tests for Supply-Chain Event Extraction.
Tests run completely offline with deterministic, explainable assertions.
"""

import json
from pathlib import Path
import pytest

from src.event_extraction import (
    EventConfig,
    EventExtractionPipeline,
    EventRecord,
    RuleBasedEventExtractor,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
)


@pytest.fixture
def extractor():
    return RuleBasedEventExtractor()


# -------------------------------------------------------------------------
# 1. Major Event Types Tests
# -------------------------------------------------------------------------

def test_extract_supply_disruption(extractor):
    record = {
        "commodity": "copper",
        "title": "Grasberg copper mine shutdown announced following workers strike",
        "url": "https://reuters.com/markets/grasberg-strike",
        "publication_timestamp": "2026-10-20T10:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.commodity == "copper"
    assert event.event_type == "supply_disruption"
    assert event.impact_direction == "potentially_bullish"
    assert event.confidence >= 0.75
    assert event.evidence_url == "https://reuters.com/markets/grasberg-strike"
    assert event.article_publication_timestamp == "2026-10-20T10:00:00Z"
    assert event.extraction_status == "success"
    assert event.evidence_fact is not None


def test_extract_demand_change_bullish(extractor):
    record = {
        "commodity": "copper",
        "title": "EV boom drives demand surge for refined copper wiring",
        "url": "https://ft.com/copper-ev",
        "publication_timestamp": "2026-10-20T11:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "demand_change"
    assert event.impact_direction == "potentially_bullish"
    assert event.extraction_status == "success"


def test_extract_demand_change_bearish(extractor):
    record = {
        "commodity": "crude oil",
        "title": "Global manufacturing slump triggers crude oil demand slowdown",
        "url": "https://bloomberg.com/oil-slump",
        "publication_timestamp": "2026-10-20T12:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "demand_change"
    assert event.impact_direction == "potentially_bearish"
    assert event.extraction_status == "success"


def test_extract_inventory_draw_bullish(extractor):
    record = {
        "commodity": "crude oil",
        "title": "EIA reports crude stockpiles drop sharply amid refinery runs",
        "url": "https://eia.gov/petroleum-report",
        "publication_timestamp": "2026-10-20T14:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "inventory_change"
    assert event.impact_direction == "potentially_bullish"
    assert event.extraction_status == "success"


def test_extract_inventory_build_bearish(extractor):
    record = {
        "commodity": "copper",
        "title": "LME copper inventories rise to multi-month high creating storage glut",
        "url": "https://lme.com/copper-inventory",
        "publication_timestamp": "2026-10-20T15:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "inventory_change"
    assert event.impact_direction == "potentially_bearish"
    assert event.extraction_status == "success"


def test_extract_production_change_cut(extractor):
    record = {
        "commodity": "crude oil",
        "title": "OPEC+ agrees to production cuts of 1 million barrels per day",
        "url": "https://reuters.com/opec-cut",
        "publication_timestamp": "2026-10-20T17:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "production_change"
    assert event.impact_direction == "potentially_bullish"
    assert event.extraction_status == "success"


def test_extract_production_change_surge(extractor):
    record = {
        "commodity": "copper",
        "title": "New mine opened in Atacama as copper production surges",
        "url": "https://mining.com/atacama",
        "publication_timestamp": "2026-10-20T18:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "production_change"
    assert event.impact_direction == "potentially_bearish"
    assert event.extraction_status == "success"


# -------------------------------------------------------------------------
# 2. Commodity-Specific Ambiguity & Nuance Tests
# -------------------------------------------------------------------------

def test_commodity_specific_geopolitical_impact(extractor):
    # Geopolitical escalation is bullish for oil (supply threat) and gold (safe haven),
    # but mixed for copper (industrial trade and global growth headwind).
    headline = "Middle East tensions escalate as military escalation looms in Gulf"

    gold_event = extractor.extract_event({"commodity": "gold", "title": headline, "url": "https://ex.com/1"})
    assert gold_event.event_type == "geopolitical_risk"
    assert gold_event.impact_direction == "potentially_bullish"

    oil_event = extractor.extract_event({"commodity": "crude oil", "title": headline, "url": "https://ex.com/2"})
    assert oil_event.event_type == "geopolitical_risk"
    assert oil_event.impact_direction == "potentially_bullish"

    copper_event = extractor.extract_event({"commodity": "copper", "title": headline, "url": "https://ex.com/3"})
    assert copper_event.event_type == "geopolitical_risk"
    assert copper_event.impact_direction == "mixed"


def test_commodity_specific_rate_cut_impact(extractor):
    # Fed rate cuts are bullish for gold (dollar/yield), but mixed for copper/oil (stimulus vs recession).
    headline = "Fed cuts rates by 50 bps sparking broad liquidity shift"

    gold_event = extractor.extract_event({"commodity": "gold", "title": headline, "url": "https://ex.com/g"})
    assert gold_event.event_type == "policy_action"
    assert gold_event.impact_direction == "potentially_bullish"

    copper_event = extractor.extract_event({"commodity": "copper", "title": headline, "url": "https://ex.com/c"})
    assert copper_event.event_type == "policy_action"
    assert copper_event.impact_direction == "mixed"


def test_commodity_specific_opec_cross_scope(extractor):
    # OPEC quotas govern crude oil. If reported under copper, direction should be unknown.
    headline = "OPEC+ agrees to production cuts of 1 million barrels"

    oil_event = extractor.extract_event({"commodity": "crude oil", "title": headline, "url": "https://ex.com/oil"})
    assert oil_event.event_type == "production_change"
    assert oil_event.impact_direction == "potentially_bullish"

    copper_event = extractor.extract_event({"commodity": "copper", "title": headline, "url": "https://ex.com/cu"})
    assert copper_event.event_type == "production_change"
    assert copper_event.impact_direction == "unknown"


# -------------------------------------------------------------------------
# 3. Negation & Cancellation Tests
# -------------------------------------------------------------------------

def test_extract_negation_strike_avoided(extractor):
    record = {
        "commodity": "copper",
        "title": "Mine strike avoided as union signs new labour agreement",
        "url": "https://reuters.com/markets/copper-strike-avoided",
        "publication_timestamp": "2026-10-20T10:30:00Z",
    }
    event = extractor.extract_event(record)

    # Disruption averted -> must not be bullish supply disruption
    assert event.event_type == "other"
    assert event.impact_direction == "unknown"
    assert event.extraction_status == "success"
    assert "negation detected" in event.evidence_fact


def test_extract_negation_production_cuts_ruled_out(extractor):
    record = {
        "commodity": "crude oil",
        "title": "OPEC production cuts ruled out at upcoming ministerial meeting",
        "url": "https://reuters.com/markets/opec-cuts-ruled-out",
        "publication_timestamp": "2026-10-20T10:45:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "other"
    assert event.impact_direction == "unknown"
    assert "negation detected" in event.evidence_fact


def test_extract_negation_strike_called_off(extractor):
    record = {
        "commodity": "copper",
        "title": "Workers call off strike at Escondida copper facility",
        "url": "https://reuters.com/markets/escondida-strike-ended",
        "publication_timestamp": "2026-10-20T11:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "other"
    assert event.impact_direction == "unknown"


# -------------------------------------------------------------------------
# 4. Misleading Keyword & Irrelevant Context Tests
# -------------------------------------------------------------------------

def test_extract_misleading_sports_gold_medal(extractor):
    record = {
        "commodity": "gold",
        "title": "Olympic athlete wins gold medal in world championship boxing tournament",
        "url": "https://sports.com/boxing-gold",
        "publication_timestamp": "2026-10-20T12:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "other"
    assert event.impact_direction == "unknown"
    assert event.confidence == 0.0
    assert event.extraction_status == "unclassified"


def test_extract_misleading_crude_joke(extractor):
    record = {
        "commodity": "crude oil",
        "title": "Comedian delivers crude joke during television broadcast interview",
        "url": "https://entertainment.com/comedy-show",
        "publication_timestamp": "2026-10-20T12:15:00Z",
    }
    event = extractor.extract_event(record)

    assert event.event_type == "other"
    assert event.impact_direction == "unknown"
    assert event.extraction_status == "unclassified"


def test_extract_conflicting_signals_mixed(extractor):
    record = {
        "commodity": "copper",
        "title": "Mine closure shuts down supply while manufacturing slump dents demand",
        "url": "https://example.com/mixed",
        "publication_timestamp": "2026-10-20T20:00:00Z",
    }
    event = extractor.extract_event(record)

    assert event.impact_direction == "mixed"
    assert event.extraction_status == "success"


# -------------------------------------------------------------------------
# 5. Missing / Invalid Fields & URL Preservation Tests
# -------------------------------------------------------------------------

def test_extract_missing_urls_and_timestamps(extractor):
    # Verify extractor never invents missing URLs or timestamps
    record = {
        "commodity": "copper",
        "title": "Mine shutdown announced following fire",
        "url": None,
        "publication_timestamp": None,
    }
    event = extractor.extract_event(record)

    assert event.event_type == "supply_disruption"
    assert event.evidence_url == ""
    assert event.article_publication_timestamp is None
    assert event.extraction_status == "success"


def test_extract_empty_or_whitespace_headline(extractor):
    for empty_title in [None, "", "   ", "\t\n"]:
        record = {
            "commodity": "copper",
            "title": empty_title,
            "url": "https://example.com/empty",
            "publication_timestamp": "2026-10-20T21:00:00Z",
        }
        event = extractor.extract_event(record)

        assert event.event_type == "other"
        assert event.impact_direction == "unknown"
        assert event.confidence == 0.0
        assert event.extraction_status == "insufficient_data"
        assert event.evidence_url == "https://example.com/empty"
        assert event.article_publication_timestamp == "2026-10-20T21:00:00Z"


def test_extract_malformed_record_type(extractor):
    for bad_input in [None, "invalid-string", 12345, []]:
        event = extractor.extract_event(bad_input)  # type: ignore
        assert event.event_type == "other"
        assert event.impact_direction == "unknown"
        assert event.extraction_status == "error"
        assert event.confidence == 0.0


def test_valid_taxonomy_constraints(extractor):
    """Ensure every event output adheres to standardized enum choices."""
    records = [
        {"commodity": "copper", "title": "Pipeline blast halts crude oil transit", "url": "https://ex.com/1"},
        {"commodity": "gold", "title": "Unrelated article headline", "url": "https://ex.com/2"},
        {"commodity": "crude oil", "title": "", "url": "https://ex.com/3"},
    ]
    for r in records:
        evt = extractor.extract_event(r)
        assert evt.event_type in VALID_EVENT_TYPES
        assert evt.impact_direction in VALID_IMPACT_DIRECTIONS
        assert 0.0 <= evt.confidence <= 1.0


def test_preservation_of_exact_processed_sentiment_schema(extractor):
    """Verify exact fixture shaped like data/processed/commodity_news_sentiment.jsonl."""
    raw_fixture = {
        "commodity": "copper",
        "title": "Bimetals : Copper Next Frontier",
        "url": "http://www.prnewswire.com/news-releases/bimetals-coppers-next-frontier-302910564.html",
        "publication_timestamp": "2026-10-15T20:15:00Z",
        "domain": "prnewswire.com",
        "language": "English",
        "source_country": "United States",
        "fetch_timestamp": "2026-10-20T13:11:20Z",
        "provider": "gdelt",
        "sentiment_label": "neutral",
        "sentiment_score": 0.8914,
        "sentiment_model": "ProsusAI/finbert",
        "sentiment_status": "success",
    }

    event = extractor.extract_event(raw_fixture)

    assert event.commodity == "copper"
    assert event.evidence_url == "http://www.prnewswire.com/news-releases/bimetals-coppers-next-frontier-302910564.html"
    assert event.article_publication_timestamp == "2026-10-15T20:15:00Z"
    assert event.event_type in VALID_EVENT_TYPES
    assert event.impact_direction in VALID_IMPACT_DIRECTIONS
    assert 0.0 <= event.confidence <= 1.0


# -------------------------------------------------------------------------
# 6. Pipeline Integration & Non-Destructive Storage Test
# -------------------------------------------------------------------------

def test_pipeline_e2e_and_non_destructive(tmp_path: Path):
    """Verify end-to-end pipeline execution and ensure source files are never altered."""
    input_file = tmp_path / "processed" / "commodity_news_sentiment.jsonl"
    input_file.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "commodity": "copper",
            "title": "Mine closure shuts down copper output in Peru",
            "url": "https://example.com/peru-mine",
            "publication_timestamp": "2026-10-20T10:00:00Z",
            "sentiment_label": "negative",
            "sentiment_score": 0.88,
        },
        {
            "commodity": "crude oil",
            "title": "EIA reports inventories drop significantly",
            "url": "https://example.com/eia-draw",
            "publication_timestamp": "2026-10-20T11:00:00Z",
            "sentiment_label": "neutral",
            "sentiment_score": 0.91,
        },
        {
            "commodity": "gold",
            "title": "City park hosts annual autumn flower show",
            "url": "https://example.com/autumn-fair",
            "publication_timestamp": "2026-10-20T12:00:00Z",
            "sentiment_label": "neutral",
            "sentiment_score": 0.95,
        },
    ]

    with input_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    initial_input_content = input_file.read_text(encoding="utf-8")

    output_file = tmp_path / "processed" / "commodity_events.jsonl"
    config = EventConfig(input_path=input_file, output_path=output_file)
    pipeline = EventExtractionPipeline(config=config)

    summary = pipeline.run(input_path=input_file, output_path=output_file)

    assert summary["success"] is True
    assert summary["total_read"] == 3
    assert summary["total_extracted"] == 3
    assert summary["event_type_counts"]["supply_disruption"] == 1
    assert summary["event_type_counts"]["inventory_change"] == 1
    assert summary["event_type_counts"]["other"] == 1

    # Verify output file
    assert output_file.exists()
    lines = [line.strip() for line in output_file.open("r", encoding="utf-8") if line.strip()]
    assert len(lines) == 3

    e0 = json.loads(lines[0])
    assert e0["commodity"] == "copper"
    assert e0["event_type"] == "supply_disruption"
    assert e0["impact_direction"] == "potentially_bullish"
    assert e0["evidence_url"] == "https://example.com/peru-mine"
    assert e0["article_publication_timestamp"] == "2026-10-20T10:00:00Z"
    assert e0["extraction_status"] == "success"

    # Verify input file was completely untouched
    assert input_file.read_text(encoding="utf-8") == initial_input_content
