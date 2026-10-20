"""
Unit tests for storage and deduplication.
"""

import json
from pathlib import Path
import pytest

from src.news_intelligence.storage import append_records, load_existing_keys


def test_append_records_creates_file_and_subdirectories(tmp_path: Path):
    target_file = tmp_path / "sub" / "dir" / "commodity_news.jsonl"
    records = [
        {
            "commodity": "copper",
            "url": "https://example.com/copper-news-1",
            "title": "Copper demand",
            "publication_timestamp": "2026-10-15T12:00:00Z",
            "domain": "example.com",
            "language": "English",
            "source_country": "United States",
            "fetch_timestamp": "2026-10-20T13:00:00Z",
            "provider": "gdelt",
        }
    ]

    saved, dups = append_records(target_file, records)
    assert saved == 1
    assert dups == 0
    assert target_file.exists()

    with target_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    saved_obj = json.loads(lines[0])
    assert saved_obj["commodity"] == "copper"
    assert saved_obj["url"] == "https://example.com/copper-news-1"


def test_append_records_deduplicates_within_same_batch(tmp_path: Path):
    target_file = tmp_path / "news.jsonl"
    records = [
        {"commodity": "copper", "url": "https://example.com/item1", "title": "Article 1"},
        {"commodity": "copper", "url": "https://example.com/item1", "title": "Article 1 duplicate"},
        {"commodity": "copper", "url": "https://example.com/item2", "title": "Article 2"},
    ]

    saved, dups = append_records(target_file, records)
    assert saved == 2
    assert dups == 1

    with target_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2


def test_append_records_deduplicates_across_sequential_runs(tmp_path: Path):
    target_file = tmp_path / "news.jsonl"
    batch1 = [
        {"commodity": "gold", "url": "https://example.com/gold-1", "title": "Gold 1"},
        {"commodity": "gold", "url": "https://example.com/gold-2", "title": "Gold 2"},
    ]
    saved1, dups1 = append_records(target_file, batch1)
    assert saved1 == 2
    assert dups1 == 0

    batch2 = [
        {"commodity": "gold", "url": "https://example.com/gold-2", "title": "Gold 2 (dup)"},
        {"commodity": "gold", "url": "https://example.com/gold-3", "title": "Gold 3"},
    ]
    # Storage should re-read existing keys automatically when not passed
    saved2, dups2 = append_records(target_file, batch2)
    assert saved2 == 1
    assert dups2 == 1

    existing_keys = load_existing_keys(target_file)
    assert len(existing_keys) == 3
    assert ("gold", "https://example.com/gold-1") in existing_keys
    assert ("gold", "https://example.com/gold-2") in existing_keys
    assert ("gold", "https://example.com/gold-3") in existing_keys


def test_distinct_commodity_same_url_allowed(tmp_path: Path):
    target_file = tmp_path / "news.jsonl"
    # An article might discuss both gold and copper
    records = [
        {"commodity": "copper", "url": "https://example.com/metals", "title": "Metals Report"},
        {"commodity": "gold", "url": "https://example.com/metals", "title": "Metals Report"},
    ]
    saved, dups = append_records(target_file, records)
    assert saved == 2
    assert dups == 0


def test_load_existing_keys_handles_corrupt_lines(tmp_path: Path):
    target_file = tmp_path / "corrupted.jsonl"
    content = (
        '{"commodity": "copper", "url": "https://example.com/valid"}\n'
        'invalid json line here\n'
        '\n'
        '{"commodity": "gold", "url": "https://example.com/valid-gold"}\n'
    )
    target_file.write_text(content, encoding="utf-8")

    keys = load_existing_keys(target_file)
    assert len(keys) == 2
    assert ("copper", "https://example.com/valid") in keys
    assert ("gold", "https://example.com/valid-gold") in keys
