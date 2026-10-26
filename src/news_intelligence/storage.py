"""
Storage module for saving and deduplicating news records in JSON Lines format (.jsonl).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def load_existing_keys(filepath: Path) -> Set[Tuple[str, str]]:
    """
    Load existing (commodity, url) keys from a JSONL file.

    Returns:
        A set of (commodity, url) tuples for O(1) deduplication lookups.
    """
    existing_keys: Set[Tuple[str, str]] = set()

    if not filepath.exists():
        return existing_keys

    try:
        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                    commodity = str(record.get("commodity", "")).strip().lower()
                    url = str(record.get("url", "")).strip()
                    if commodity and url:
                        existing_keys.add((commodity, url))
                except json.JSONDecodeError as err:
                    logger.warning(
                        "Skipping invalid JSON on line %d of %s: %s",
                        line_num,
                        filepath,
                        err,
                    )
    except Exception as err:
        logger.error("Error reading existing keys from %s: %s", filepath, err)

    logger.debug(
        "Loaded %d existing deduplication keys from %s",
        len(existing_keys),
        filepath,
    )
    return existing_keys


def append_records(
    filepath: Path,
    records: Iterable[Dict[str, Any]],
    existing_keys: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[int, int]:
    """
    Append normalized article records to a JSONL file, deduplicating by (commodity, url).

    Args:
        filepath: Target Path to the .jsonl file.
        records: Iterable of normalized article dictionaries.
        existing_keys: Optional set of already loaded (commodity, url) keys.
                       If None, will be loaded from filepath automatically.

    Returns:
        Tuple of (saved_count, duplicate_count).
    """
    if existing_keys is None:
        existing_keys = load_existing_keys(filepath)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    duplicate_count = 0

    with filepath.open("a", encoding="utf-8") as f:
        for record in records:
            if not isinstance(record, dict):
                continue

            commodity = str(record.get("commodity", "")).strip().lower()
            url = str(record.get("url", "")).strip()

            if not commodity or not url:
                logger.warning("Skipping record missing commodity or url: %s", record)
                continue

            key = (commodity, url)
            if key in existing_keys:
                duplicate_count += 1
                logger.debug("Duplicate skipped for %s: %s", commodity, url)
                continue

            line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")
            existing_keys.add(key)
            saved_count += 1

    logger.info(
        "Appended %d new records to %s (%d duplicates skipped).",
        saved_count,
        filepath,
        duplicate_count,
    )
    return saved_count, duplicate_count
