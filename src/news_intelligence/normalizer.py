"""
Article normalization module for converting raw GDELT responses into standardized records.
"""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def parse_gdelt_timestamp(raw_date: Optional[str]) -> Optional[str]:
    """
    Parse a GDELT timestamp (typically 'YYYYMMDDTHHMMSSZ' or 'YYYYMMDDHHMMSS')
    into a standardized ISO 8601 UTC string: 'YYYY-MM-DDTHH:MM:SSZ'.

    Returns None if raw_date is missing, empty, or cannot be parsed.
    """
    if not raw_date or not isinstance(raw_date, str):
        return None

    cleaned = raw_date.strip()
    if not cleaned:
        return None

    # Common GDELT format: '20261015T201500Z'
    try:
        dt = datetime.strptime(cleaned, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    # Alternative GDELT format without T/Z: '20261015201500'
    try:
        dt = datetime.strptime(cleaned, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    # Alternative GDELT date only: '20261015'
    try:
        dt = datetime.strptime(cleaned, "%Y%m%d").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        pass

    # Standard ISO format fallback: '2026-10-15T20:15:00Z' or with offset
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    logger.debug("Could not parse GDELT date string: '%s'", raw_date)
    return None


def extract_domain_from_url(url: str) -> str:
    """Extract domain host name from a URL string."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        # Strip port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Strip leading 'www.'
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def normalize_article(
    raw_article: Dict[str, Any],
    commodity: str,
    fetch_timestamp: Optional[str] = None,
    provider: str = "gdelt",
) -> Optional[Dict[str, Any]]:
    """
    Normalize a raw GDELT article dictionary into the standard schema.

    Schema:
        commodity: str (e.g. 'copper', 'gold', 'crude oil')
        title: str (stripped headline or empty string)
        url: str (validated article URL)
        publication_timestamp: Optional[str] (ISO 8601 UTC string or None)
        domain: str (news source domain)
        language: str (language identifier, e.g. 'English', or 'unknown')
        source_country: Optional[str] (country of source or None)
        fetch_timestamp: str (ISO 8601 UTC timestamp of fetch)
        provider: str ('gdelt')

    Returns:
        Normalized dictionary, or None if the article lacks a valid URL.
    """
    if not isinstance(raw_article, dict):
        return None

    # Validate and clean URL (support 'url' or 'URL')
    raw_url = raw_article.get("url") or raw_article.get("URL")
    if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
        logger.debug("Skipping article without valid URL: %s", raw_article)
        return None
    url = raw_url.strip()

    # Validate and clean Title
    raw_title = raw_article.get("title") or raw_article.get("Title") or ""
    title = str(raw_title).strip() if raw_title else ""

    # Parse publication timestamp (from GDELT 'seendate' or 'seendate' variants)
    raw_date = raw_article.get("seendate") or raw_article.get("publication_timestamp")
    publication_timestamp = parse_gdelt_timestamp(str(raw_date)) if raw_date else None

    # Domain
    raw_domain = raw_article.get("domain") or raw_article.get("Domain")
    if raw_domain and isinstance(raw_domain, str) and raw_domain.strip():
        domain = raw_domain.strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
    else:
        domain = extract_domain_from_url(url)

    # Language
    raw_lang = raw_article.get("language") or raw_article.get("Language")
    if raw_lang and isinstance(raw_lang, str) and raw_lang.strip():
        language = raw_lang.strip()
    else:
        language = "unknown"

    # Source Country
    raw_country = (
        raw_article.get("sourcecountry")
        or raw_article.get("source_country")
        or raw_article.get("SourceCountry")
    )
    if raw_country and isinstance(raw_country, str) and raw_country.strip():
        source_country: Optional[str] = raw_country.strip()
    else:
        source_country = None

    # Fetch timestamp (default to current UTC ISO-8601)
    if not fetch_timestamp:
        fetch_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "commodity": commodity.strip().lower(),
        "title": title,
        "url": url,
        "publication_timestamp": publication_timestamp,
        "domain": domain,
        "language": language,
        "source_country": source_country,
        "fetch_timestamp": fetch_timestamp,
        "provider": provider,
    }
