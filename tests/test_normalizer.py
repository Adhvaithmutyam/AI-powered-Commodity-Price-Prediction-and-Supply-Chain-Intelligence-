"""
Unit tests for article normalization and date parsing.
"""

from src.news_intelligence.normalizer import (
    extract_domain_from_url,
    normalize_article,
    parse_gdelt_timestamp,
)


def test_parse_gdelt_timestamp_standard_format():
    assert parse_gdelt_timestamp("20261015T201500Z") == "2026-10-15T20:15:00Z"


def test_parse_gdelt_timestamp_digits_only():
    assert parse_gdelt_timestamp("20261015201500") == "2026-10-15T20:15:00Z"


def test_parse_gdelt_timestamp_date_only():
    assert parse_gdelt_timestamp("20261015") == "2026-10-15T00:00:00Z"


def test_parse_gdelt_timestamp_iso_format():
    assert parse_gdelt_timestamp("2026-10-15T20:15:00Z") == "2026-10-15T20:15:00Z"


def test_parse_gdelt_timestamp_invalid_or_none():
    assert parse_gdelt_timestamp(None) is None
    assert parse_gdelt_timestamp("") is None
    assert parse_gdelt_timestamp("not-a-date") is None


def test_extract_domain_from_url():
    assert extract_domain_from_url("https://www.reuters.com/markets/commodities") == "reuters.com"
    assert extract_domain_from_url("http://bloomberg.com:8080/news") == "bloomberg.com"
    assert extract_domain_from_url("invalid-url") == ""


def test_normalize_article_complete():
    raw = {
        "url": "https://www.ft.com/content/12345",
        "title": "Copper rallies as supply tightens",
        "seendate": "20261015T120000Z",
        "domain": "ft.com",
        "language": "English",
        "sourcecountry": "United Kingdom",
    }
    normalized = normalize_article(
        raw,
        commodity="copper",
        fetch_timestamp="2026-10-20T13:00:00Z",
    )

    assert normalized == {
        "commodity": "copper",
        "title": "Copper rallies as supply tightens",
        "url": "https://www.ft.com/content/12345",
        "publication_timestamp": "2026-10-15T12:00:00Z",
        "domain": "ft.com",
        "language": "English",
        "source_country": "United Kingdom",
        "fetch_timestamp": "2026-10-20T13:00:00Z",
        "provider": "gdelt",
    }


def test_normalize_article_missing_url_returns_none():
    raw_no_url = {"title": "Gold price falls", "seendate": "20261015T120000Z"}
    assert normalize_article(raw_no_url, commodity="gold") is None

    raw_empty_url = {"url": "   ", "title": "Gold price falls"}
    assert normalize_article(raw_empty_url, commodity="gold") is None


def test_normalize_article_missing_optional_fields():
    raw = {
        "url": "https://www.example.com/crude-oil-update",
    }
    normalized = normalize_article(
        raw,
        commodity="crude oil",
        fetch_timestamp="2026-10-20T13:00:00Z",
    )

    assert normalized is not None
    assert normalized["commodity"] == "crude oil"
    assert normalized["title"] == ""
    assert normalized["url"] == "https://www.example.com/crude-oil-update"
    assert normalized["publication_timestamp"] is None
    assert normalized["domain"] == "example.com"
    assert normalized["language"] == "unknown"
    assert normalized["source_country"] is None
    assert normalized["fetch_timestamp"] == "2026-10-20T13:00:00Z"
    assert normalized["provider"] == "gdelt"


def test_normalize_article_whitespace_stripping():
    raw = {
        "url": "  https://example.com/gold   ",
        "title": "   Gold prices steady   ",
        "sourcecountry": "   United States   ",
        "domain": "  www.EXAMPLE.com  ",
    }
    normalized = normalize_article(raw, commodity="  GOLD  ")
    assert normalized is not None
    assert normalized["commodity"] == "gold"
    assert normalized["url"] == "https://example.com/gold"
    assert normalized["title"] == "Gold prices steady"
    assert normalized["domain"] == "example.com"
    assert normalized["source_country"] == "United States"
