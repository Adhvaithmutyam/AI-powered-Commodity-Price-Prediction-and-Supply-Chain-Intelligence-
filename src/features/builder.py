"""
Article-level NLP Feature Builder.

Joins raw news, FinBERT sentiment, explainable relevance, and supply-chain event
outputs into a standardized, traceable article-level feature record.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.news_intelligence.relevance import assess_relevance

logger = logging.getLogger(__name__)


def canonicalize_url(url: Optional[str]) -> str:
    """
    Standardize a URL for reliable article identity joining.
    Strips whitespace and normalizes trailing slashes.
    """
    if not url or not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if cleaned.endswith("/") and len(cleaned.split("/")) > 4:
        cleaned = cleaned[:-1]
    return cleaned


def _extract_matched_indicators(raw_val: Any) -> List[str]:
    """
    Safely extract and normalize matched relevance indicators into a list of strings.
    Handles None, missing, string, list, set, tuple, and malformed types.
    """
    if raw_val is None:
        return []
    if isinstance(raw_val, (list, tuple, set)):
        indicators: List[str] = []
        for item in raw_val:
            if item is not None:
                val_str = str(item).strip()
                if val_str:
                    indicators.append(val_str)
        return indicators
    if isinstance(raw_val, str):
        cleaned = raw_val.strip()
        return [cleaned] if cleaned else []
    return []


def load_jsonl_records(filepath: Path) -> List[Dict[str, Any]]:
    """
    Safely load JSON objects from a JSONL file.
    Skips empty lines and logs warnings for malformed lines.
    """
    if not filepath.exists():
        logger.info("File does not exist: %s", filepath)
        return []

    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                if isinstance(data, dict):
                    records.append(data)
                else:
                    logger.warning("Line %d in %s is not a JSON object", line_num, filepath)
            except json.JSONDecodeError as err:
                logger.warning("Malformed JSON on line %d in %s: %s", line_num, filepath, err)
    return records


class ArticleFeatureBuilder:
    """
    Constructs standardized article-level NLP features by joining sentiment,
    relevance, and event datasets by (commodity, canonical_url).
    """

    def __init__(self) -> None:
        pass

    def build(
        self,
        news_records: Optional[List[Dict[str, Any]]] = None,
        sentiment_records: Optional[List[Dict[str, Any]]] = None,
        event_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Join and normalize records into standardized article-level NLP features.

        Args:
            news_records: Raw or normalized article records.
            sentiment_records: FinBERT sentiment-enriched records.
            event_records: Extracted supply-chain event records.

        Returns:
            List of standardized article-level feature dictionaries.
        """
        # 1. Map event records by (commodity, canonical_url)
        event_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if event_records:
            for ev in event_records:
                comm = str(ev.get("commodity", "")).strip().lower()
                url = canonicalize_url(ev.get("evidence_url") or ev.get("url"))
                if comm and url:
                    # Keep first matching event record for key
                    if (comm, url) not in event_map:
                        event_map[(comm, url)] = ev

        # 2. Map sentiment records by (commodity, canonical_url)
        sentiment_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if sentiment_records:
            for sent in sentiment_records:
                comm = str(sent.get("commodity", "")).strip().lower()
                url = canonicalize_url(sent.get("url") or sent.get("evidence_url"))
                if comm and url:
                    # Keep first matching sentiment record for key
                    if (comm, url) not in sentiment_map:
                        sentiment_map[(comm, url)] = sent

        seen_keys: Set[Tuple[str, str]] = set()
        merged_articles: List[Dict[str, Any]] = []

        # 3. Process raw news records (if provided)
        # Every unique raw article is processed; matching sentiment records are merged
        if news_records:
            for raw_rec in news_records:
                comm = str(raw_rec.get("commodity", "")).strip().lower()
                url = canonicalize_url(raw_rec.get("url") or raw_rec.get("evidence_url"))
                if not comm or not url:
                    continue
                key = (comm, url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Check if matching sentiment record exists
                sent_rec = sentiment_map.get(key)
                if sent_rec:
                    # Base metadata from raw news, enriched with sentiment record fields taking precedence
                    base_rec = dict(raw_rec)
                    for k, v in sent_rec.items():
                        if v is not None or k not in base_rec:
                            base_rec[k] = v
                    has_sentiment = (
                        "sentiment_label" in sent_rec
                        and sent_rec.get("sentiment_status") != "missing"
                        and sent_rec.get("sentiment_label") is not None
                    )
                else:
                    base_rec = raw_rec
                    has_sentiment = (
                        "sentiment_label" in raw_rec
                        and raw_rec.get("sentiment_status") != "missing"
                        and raw_rec.get("sentiment_label") is not None
                    )

                ev_data = event_map.get(key)
                feature_row = self._create_feature_record(
                    base_record=base_rec,
                    event_data=ev_data,
                    has_sentiment=has_sentiment,
                )
                merged_articles.append(feature_row)

        # 4. Process sentiment-only records (articles in sentiment_records not present in news_records)
        if sentiment_records:
            for sent_rec in sentiment_records:
                comm = str(sent_rec.get("commodity", "")).strip().lower()
                url = canonicalize_url(sent_rec.get("url") or sent_rec.get("evidence_url"))
                if not comm or not url:
                    continue
                key = (comm, url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                has_sentiment = (
                    "sentiment_label" in sent_rec
                    and sent_rec.get("sentiment_status") != "missing"
                    and sent_rec.get("sentiment_label") is not None
                )
                ev_data = event_map.get(key)
                feature_row = self._create_feature_record(
                    base_record=sent_rec,
                    event_data=ev_data,
                    has_sentiment=has_sentiment,
                )
                merged_articles.append(feature_row)

        # 5. Process event-only records (articles in event_records not present in news or sentiment)
        if event_records:
            for ev in event_records:
                comm = str(ev.get("commodity", "")).strip().lower()
                url = canonicalize_url(ev.get("evidence_url") or ev.get("url"))
                if not comm or not url:
                    continue
                key = (comm, url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                base_rec = {
                    "commodity": comm,
                    "url": url,
                    "title": ev.get("event_summary", ""),
                    "publication_timestamp": ev.get("article_publication_timestamp"),
                }
                feature_row = self._create_feature_record(
                    base_record=base_rec,
                    event_data=ev,
                    has_sentiment=False,
                )
                merged_articles.append(feature_row)

        return merged_articles

    def _create_feature_record(
        self,
        base_record: Dict[str, Any],
        event_data: Optional[Dict[str, Any]],
        has_sentiment: bool,
    ) -> Dict[str, Any]:
        """Combine an individual article's components into the standard schema."""
        comm = str(base_record.get("commodity", "")).strip().lower()
        url = canonicalize_url(base_record.get("url") or base_record.get("evidence_url"))
        title = str(base_record.get("title") or base_record.get("event_summary") or "").strip()
        pub_ts = base_record.get("publication_timestamp") or base_record.get("article_publication_timestamp")
        domain = base_record.get("domain") or ""
        language = base_record.get("language") or "unknown"
        source_country = base_record.get("source_country")
        fetch_ts = base_record.get("fetch_timestamp")
        provider = base_record.get("provider") or "gdelt"

        # 1. Relevance Features:
        # Check if already present in base_record
        if "relevance_class" in base_record and "relevance_score" in base_record:
            rel_class = base_record.get("relevance_class") or "uncertain"
            raw_rel_score = base_record.get("relevance_score")
            try:
                rel_score = float(raw_rel_score) if raw_rel_score is not None else 0.0
            except (ValueError, TypeError):
                rel_score = 0.0
            rel_explanation = base_record.get("relevance_explanation") or ""
            raw_indicators = base_record.get("matched_indicators")
            if raw_indicators is None:
                raw_indicators = base_record.get("matched_relevance_indicators")
            matched_indicators = _extract_matched_indicators(raw_indicators)
        elif title:
            # Dynamically compute explainable relevance
            rel_assessment = assess_relevance(
                headline=title,
                commodity=comm,
                language=language,
                url=url,
            )
            rel_class = rel_assessment.relevance_class
            rel_score = rel_assessment.relevance_score
            rel_explanation = rel_assessment.relevance_explanation
            matched_indicators = _extract_matched_indicators(rel_assessment.matched_indicators)
        else:
            rel_class = "uncertain"
            rel_score = 0.40
            rel_explanation = "Title is missing; relevance cannot be assessed from headline alone."
            matched_indicators = ["missing_headline"]

        # 2. Sentiment Features:
        if has_sentiment:
            sentiment_label = base_record.get("sentiment_label")
            raw_sent_score = base_record.get("sentiment_score")
            sentiment_score = float(raw_sent_score) if raw_sent_score is not None else None
            sentiment_model = base_record.get("sentiment_model", "ProsusAI/finbert")
            sentiment_status = base_record.get("sentiment_status", "success")
        else:
            sentiment_label = None
            sentiment_score = None
            sentiment_model = None
            sentiment_status = "missing"

        # 3. Supply-Chain Event Features:
        if event_data:
            event_type = event_data.get("event_type", "other")
            event_impact_direction = event_data.get("impact_direction", "unknown")
            raw_conf = event_data.get("confidence")
            event_confidence = float(raw_conf) if raw_conf is not None else 0.0
            extraction_status = event_data.get("extraction_status", "success")
            evidence_fact = event_data.get("evidence_fact")
            implication_rationale = event_data.get("implication_rationale")
        else:
            event_type = "other"
            event_impact_direction = "unknown"
            event_confidence = 0.0
            extraction_status = "missing"
            evidence_fact = None
            implication_rationale = None

        # 4. Overall Feature Completeness Status
        if has_sentiment and event_data:
            feature_status = "complete"
        elif has_sentiment and not event_data:
            feature_status = "partial_missing_events"
        elif not has_sentiment and event_data:
            feature_status = "partial_missing_sentiment"
        else:
            feature_status = "partial_missing_both"

        return {
            "commodity": comm,
            "article_url": url,
            "article_title": title,
            "article_publication_timestamp": pub_ts,
            "domain": domain,
            "language": language,
            "source_country": source_country,
            "fetch_timestamp": fetch_ts,
            "provider": provider,
            "relevance_class": rel_class,
            "relevance_score": rel_score,
            "relevance_explanation": rel_explanation,
            "matched_relevance_indicators": matched_indicators,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "sentiment_model": sentiment_model,
            "sentiment_status": sentiment_status,
            "event_type": event_type,
            "event_impact_direction": event_impact_direction,
            "event_confidence": event_confidence,
            "extraction_status": extraction_status,
            "evidence_fact": evidence_fact,
            "implication_rationale": implication_rationale,
            "feature_status": feature_status,
        }
