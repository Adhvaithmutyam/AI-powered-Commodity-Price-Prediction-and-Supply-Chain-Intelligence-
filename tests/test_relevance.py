"""
Unit and integration tests for news relevance assessment and query construction.
All tests use offline fixtures and do not make external network calls.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.news_intelligence.config import PipelineConfig
from src.news_intelligence.gdelt_client import GDELTClient
from src.news_intelligence.pipeline import NewsIngestionPipeline
from src.news_intelligence.relevance import (
    RelevanceClass,
    RelevanceResult,
    assess_relevance,
    enrich_record_with_relevance,
)


# ---------------------------------------------------------------------------
# Step 4 Tests: Directly Relevant Commodity Headlines
# ---------------------------------------------------------------------------

def test_directly_relevant_copper_headlines():
    """Verify operational, price, and supply-chain copper headlines are classified directly relevant."""
    samples = [
        "Freeport declares force majeure at Grasberg copper mine after heavy flooding",
        "Workers launch indefinite strike at Chilean Escondida copper mine",
        "Chinese smelters agree on 10% copper production cuts amid concentrate shortage",
        "LME copper inventories plunge to lowest levels since 2021",
        "Copper price surges to 14-month high on global renewable energy demand",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="copper")
        assert res.relevance_class == RelevanceClass.DIRECTLY_RELEVANT.value, f"Failed on: {headline}"
        assert res.relevance_score >= 0.80
        assert "commodity_copper" in res.matched_indicators or "mining_production" in res.matched_indicators
        assert "headline analysis" in res.relevance_explanation.lower()


def test_directly_relevant_gold_headlines():
    """Verify bullion, mining, central bank, and price gold headlines are classified directly relevant."""
    samples = [
        "Central banks accelerate bullion purchases to record gold reserves",
        "Spot gold price reaches all-time high as geopolitical tensions flare",
        "Gold futures climb past $2,700 an ounce on persistent inflation concerns",
        "Barrick Gold expands gold mining operations in Nevada complex",
        "Investors pour capital into physical gold ETFs amid equity market volatility",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="gold")
        assert res.relevance_class == RelevanceClass.DIRECTLY_RELEVANT.value, f"Failed on: {headline}"
        assert res.relevance_score >= 0.80
        assert "commodity_gold" in res.matched_indicators
        assert len(res.matched_indicators) >= 2


def test_directly_relevant_crude_oil_headlines():
    """Verify OPEC, barrels, refinery, and energy oil headlines are classified directly relevant."""
    samples = [
        "OPEC+ agrees to extend voluntary crude oil production cuts of 2.2 million bpd",
        "Brent crude rises toward $90 a barrel following Red Sea tanker disruptions",
        "EIA reports unexpected crude oil inventory draw of 4.5 million barrels",
        "Refinery fire in Houston halts petroleum distillation operations",
        "US crude oil output hits record 13.3 million barrels per day",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="crude oil")
        assert res.relevance_class == RelevanceClass.DIRECTLY_RELEVANT.value, f"Failed on: {headline}"
        assert res.relevance_score >= 0.80
        assert "commodity_crude oil" in res.matched_indicators


# ---------------------------------------------------------------------------
# Step 4 Tests: Potentially Relevant Macroeconomic Drivers
# ---------------------------------------------------------------------------

def test_potentially_relevant_macro_monetary_policy():
    """Verify broad central bank and interest rate headlines without commodity names are potentially relevant."""
    samples = [
        "Federal Reserve cuts benchmark interest rate by 50 basis points",
        "ECB signals further monetary policy easing as European growth slows",
        "Fed Chair Jerome Powell hints at cautious pace of future rate cuts",
        "Bank of Japan raises interest rates amid rising wage inflation",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="copper")
        assert res.relevance_class == RelevanceClass.POTENTIALLY_RELEVANT.value, f"Failed on: {headline}"
        assert 0.60 <= res.relevance_score <= 0.80
        assert "monetary_policy" in res.matched_indicators


def test_potentially_relevant_macro_inflation_data():
    """Verify inflation indices (CPI, PPI) are classified as potentially relevant."""
    samples = [
        "US CPI report reveals consumer prices cooled to 2.4% annual rate",
        "Producer price index (PPI) rises more than expected in September",
        "Stagflation fears re-emerge as core inflation proves sticky",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="gold")
        assert res.relevance_class == RelevanceClass.POTENTIALLY_RELEVANT.value, f"Failed on: {headline}"
        assert "inflation_data" in res.matched_indicators


def test_potentially_relevant_macro_currency_and_yields():
    """Verify dollar index (DXY) and treasury yield moves are potentially relevant."""
    samples = [
        "US Dollar Index (DXY) rallies to fresh monthly highs after strong jobs report",
        "10-year Treasury yields drop sharply as bond investors seek safety",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="crude oil")
        assert res.relevance_class == RelevanceClass.POTENTIALLY_RELEVANT.value, f"Failed on: {headline}"
        assert "currency_yields" in res.matched_indicators


def test_potentially_relevant_macro_pmi_and_trade():
    """Verify manufacturing PMI, tariffs, and trade war headlines are potentially relevant."""
    samples = [
        "Global manufacturing PMI contracts to 48.8 signaling weakened industrial demand",
        "New trade tariffs announced on industrial raw materials and metals",
        "China announces massive fiscal stimulus package to revive factory activity",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="copper")
        assert res.relevance_class == RelevanceClass.POTENTIALLY_RELEVANT.value, f"Failed on: {headline}"
        assert any(m in res.matched_indicators for m in ["macro_growth_pmi", "trade_geopolitics"])


def test_potentially_relevant_unspecified_operational():
    """Verify operational supply signals without specific commodity name are potentially relevant."""
    headline = "Major refinery outage halts commercial fuel exports"
    res = assess_relevance(headline, commodity="copper")
    assert res.relevance_class == RelevanceClass.POTENTIALLY_RELEVANT.value
    assert "mining_production" in res.matched_indicators


# ---------------------------------------------------------------------------
# Step 4 Tests: Irrelevant Noise, Sports, and Misleading Homonyms
# ---------------------------------------------------------------------------

def test_irrelevant_sports_competitions():
    """Verify Olympic and sporting medal headlines are classified irrelevant."""
    samples = [
        "Team USA athlete wins gold medal in Olympic 100m hurdles final",
        "Swimmer claims silver medal at World Championship in Budapest",
        "Olympic medalist celebrates victory on the podium in Paris",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="gold")
        assert res.relevance_class == RelevanceClass.IRRELEVANT.value, f"Failed on: {headline}"
        assert res.relevance_score <= 0.20
        assert "sports_competition" in res.matched_indicators


def test_irrelevant_figurative_and_slang():
    """Verify figurative phrases ('crude joke', 'golden age', 'golden retriever') are irrelevant."""
    samples = [
        "Stand-up comedian delivers crude joke about Hollywood celebrities",
        "Hollywood enters a golden age of prestige television drama",
        "Family adopts golden retriever puppy from local animal rescue",
        "CEO receives golden handshake severance package upon retirement",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="crude oil")
        assert res.relevance_class == RelevanceClass.IRRELEVANT.value, f"Failed on: {headline}"
        assert res.relevance_score <= 0.20
        assert "idiomatic_figurative" in res.matched_indicators


def test_irrelevant_consumer_goods():
    """Verify domestic household goods (copper pipes, gold earrings) are irrelevant."""
    samples = [
        "Contractor fixes leaking copper pipes in residential basement",
        "Jewelry boutique unveils handmade gold earrings for holiday shoppers",
    ]
    for headline in samples:
        res = assess_relevance(headline, commodity="copper")
        assert res.relevance_class == RelevanceClass.IRRELEVANT.value, f"Failed on: {headline}"
        assert "consumer_household_goods" in res.matched_indicators


def test_irrelevant_foreign_currency_homonyms():
    """Verify Vietnamese currency ('đồng') homonym is flagged as noise."""
    headline = "Đắk Lắk : Giáo sư được hỗ trợ 2 tỷ , thạc sĩ 700 triệu đồng"
    res = assess_relevance(headline, commodity="copper", language="Vietnamese")
    assert res.relevance_class == RelevanceClass.IRRELEVANT.value
    assert "foreign_homonyms" in res.matched_indicators


def test_irrelevant_unrelated_venue_announcement():
    """Verify completely unrelated headlines from search matches are classified irrelevant."""
    headline = "New Venues in Boston for Fall 2026 Meetings and Events"
    res = assess_relevance(headline, commodity="gold")
    assert res.relevance_class == RelevanceClass.IRRELEVANT.value
    assert "no_domain_signals" in res.matched_indicators


# ---------------------------------------------------------------------------
# Step 4 Tests: Uncertain and Edge Cases
# ---------------------------------------------------------------------------

def test_uncertain_empty_and_missing_headlines():
    """Verify empty, None, and whitespace headlines return uncertain with 0.40 score."""
    for empty_val in [None, "", "   ", "\t\n"]:
        res = assess_relevance(empty_val, commodity="copper")
        assert res.relevance_class == RelevanceClass.UNCERTAIN.value
        assert res.relevance_score == 0.40
        assert "missing_headline" in res.matched_indicators


def test_uncertain_generic_boilerplate_headlines():
    """Verify uninformative boilerplate headlines return uncertain."""
    for bp in ["Press Release", "Update", "News", "Alert"]:
        res = assess_relevance(bp, commodity="gold")
        assert res.relevance_class == RelevanceClass.UNCERTAIN.value
        assert "boilerplate_headline" in res.matched_indicators


def test_uncertain_non_english_without_english_domain_terms():
    """Verify non-English text lacking English operational terms returns uncertain."""
    headline = "Dự án bò Bình Hà liên quan đại án Trần Bắc Hà"
    res = assess_relevance(headline, commodity="copper", language="Vietnamese")
    assert res.relevance_class == RelevanceClass.UNCERTAIN.value
    assert "non_english_vietnamese" in res.matched_indicators


# ---------------------------------------------------------------------------
# Step 4 Tests: Schema Preservation and Pipeline Integration
# ---------------------------------------------------------------------------

def test_enrich_record_with_relevance_preserves_all_fields():
    """Verify enrich_record_with_relevance appends fields without modifying raw input."""
    raw_record = {
        "commodity": "copper",
        "title": "Chilean copper mine strike threatens global supply",
        "url": "https://reuters.com/copper-strike",
        "publication_timestamp": "2026-10-20T10:00:00Z",
        "domain": "reuters.com",
        "language": "English",
        "source_country": "Chile",
        "fetch_timestamp": "2026-10-20T11:00:00Z",
        "provider": "gdelt",
    }
    enriched = enrich_record_with_relevance(raw_record)

    # All 9 original fields preserved exactly
    for k, v in raw_record.items():
        assert enriched[k] == v

    # 4 new relevance fields present
    assert enriched["relevance_class"] == RelevanceClass.DIRECTLY_RELEVANT.value
    assert enriched["relevance_score"] >= 0.85
    assert isinstance(enriched["relevance_explanation"], str)
    assert isinstance(enriched["matched_indicators"], list)
    assert "supply_disruption" in enriched["matched_indicators"]


def test_query_construction_domain_terms_and_filters():
    """Verify PipelineConfig commodity_queries use targeted terms, negative keywords, and English filter."""
    config = PipelineConfig()
    queries = config.commodity_queries

    # Copper: must include domain terms and English filter
    assert "copper" in queries
    assert "mining" in queries["copper"]
    assert "smelter" in queries["copper"]
    assert "sourcelang:english" in queries["copper"]

    # Gold: must include bullion/price, negative sports keywords, and English filter
    assert "gold" in queries
    assert "bullion" in queries["gold"]
    assert "-olympic" in queries["gold"]
    assert '-"gold medal"' in queries["gold"]
    assert "sourcelang:english" in queries["gold"]

    # Crude Oil: must include OPEC, barrels, refinery, and English filter
    assert "crude oil" in queries
    assert "OPEC" in queries["crude oil"]
    assert "barrels" in queries["crude oil"]
    assert "refinery" in queries["crude oil"]
    assert "sourcelang:english" in queries["crude oil"]


def test_pipeline_relevance_enrichment_integration(tmp_path: Path):
    """Verify NewsIngestionPipeline with assess_relevance=True records relevance stats."""
    output_file = tmp_path / "commodity_news.jsonl"
    config = PipelineConfig(
        commodities=["copper", "gold"],
        output_path=output_file,
        rate_limit_delay_seconds=0.0,
    )

    mock_client = MagicMock(spec=GDELTClient)
    mock_client.fetch_articles.side_effect = lambda query, **kwargs: (
        [
            {
                "title": "Chilean copper mine strike halts operations",
                "url": "https://news.com/copper-strike",
                "seendate": "20261020T120000Z",
                "language": "English",
            }
        ]
        if "copper" in query
        else [
            {
                "title": "Olympic champion wins gold medal in sprint",
                "url": "https://news.com/gold-medal",
                "seendate": "20261020T120000Z",
                "language": "English",
            }
        ]
    )

    pipeline = NewsIngestionPipeline(config=config, client=mock_client)
    summary = pipeline.run(output_path=output_file, assess_relevance=True)

    assert summary["success"] is True
    assert summary["total_saved"] == 2
    assert "relevance_counts" in summary
    assert summary["relevance_counts"]["directly_relevant"] == 1
    assert summary["relevance_counts"]["irrelevant"] == 1
