"""
Data Quality Validation Module for NLP Feature Datasets.

Performs comprehensive audits of article-level NLP features, detecting schema anomalies,
duplicate identities, invalid timestamps, future-date violations, missing pipeline stages,
and out-of-bounds confidence values.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from src.features.builder import canonicalize_url, load_jsonl_records
from src.features.aggregator import parse_timestamp_utc
from src.evaluation.config import (
    EvaluationConfig,
    VALID_COMMODITIES,
    VALID_RELEVANCE_CLASSES,
    VALID_SENTIMENT_LABELS,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
    VALID_FEATURE_STATUSES,
)

logger = logging.getLogger(__name__)


def _is_valid_url(url: Any) -> bool:
    """Check if a URL is a non-empty string with http/https scheme and a netloc."""
    if not url or not isinstance(url, str):
        return False
    cleaned = url.strip()
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        return False
    try:
        parsed = urlparse(cleaned)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def _is_valid_confidence(val: Any) -> bool:
    """Check if a value is a finite float/int within the inclusive range [0.0, 1.0]."""
    if val is None:
        return False
    if isinstance(val, bool):
        return False
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return False
        return 0.0 <= f_val <= 1.0
    except (ValueError, TypeError):
        return False


class DataQualityValidator:
    """
    Independent validator for standardized article-level NLP feature datasets.
    Audits structural integrity, distribution balance, and temporal sanity.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    def validate(
        self,
        records: List[Dict[str, Any]],
        evaluation_cutoff: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Validate a collection of article-level feature records.

        Args:
            records: List of article feature dictionaries.
            evaluation_cutoff: Optional UTC datetime cutoff to detect future-dated articles.
                               If None, defaults to current UTC time.

        Returns:
            Structured dictionary summarizing all data-quality metrics and anomalies.
        """
        total_records = len(records)
        now_utc = evaluation_cutoff or datetime.now(timezone.utc)
        cutoff_iso = now_utc.isoformat()

        # Commodity counts
        commodity_counts: Dict[str, int] = {comm: 0 for comm in self.config.commodities}
        unsupported_commodities: Counter = Counter()

        # Identity & Deduplication tracking
        seen_identities: Set[Tuple[str, str]] = set()
        duplicate_identities: List[Dict[str, Any]] = []

        # URL validation tracking
        missing_urls: int = 0
        invalid_urls: List[Dict[str, Any]] = []

        # Timestamp validation tracking
        missing_timestamps: int = 0
        unparseable_timestamps: List[Dict[str, Any]] = []
        future_timestamps: List[Dict[str, Any]] = []
        valid_timestamps: int = 0

        # Confidence bounds tracking
        invalid_relevance_scores: List[Dict[str, Any]] = []
        invalid_sentiment_scores: List[Dict[str, Any]] = []
        invalid_event_confidences: List[Dict[str, Any]] = []

        # Distribution counters
        relevance_distribution: Counter = Counter()
        sentiment_distribution: Counter = Counter()
        sentiment_status_distribution: Counter = Counter()
        event_type_distribution: Counter = Counter()
        impact_direction_distribution: Counter = Counter()
        extraction_status_distribution: Counter = Counter()
        feature_status_distribution: Counter = Counter()

        # Taxonomy violations
        unsupported_relevance_classes: Counter = Counter()
        unsupported_sentiment_labels: Counter = Counter()
        unsupported_event_types: Counter = Counter()
        unsupported_impact_directions: Counter = Counter()
        unsupported_feature_statuses: Counter = Counter()

        # Missingness & pipeline status counts
        missing_sentiment_count: int = 0
        missing_event_extraction_count: int = 0
        genuine_absence_of_events_count: int = 0  # Processed, but no event matched (event_type == 'other')

        # Malformed records tracking
        malformed_records: List[Dict[str, Any]] = []

        # Track unique indices of records with any validation defect
        # Ensures multi-failure records are not double-counted when computing pass rate
        invalid_record_indices: Set[int] = set()

        # Detailed pipeline status counters
        successful_sentiment_count: int = 0
        failed_sentiment_count: int = 0
        successful_event_extraction_count: int = 0
        failed_event_extraction_count: int = 0
        no_event_rule_match_count: int = 0  # Processed by rules, but no pattern matched (event_type == 'other')

        for idx, rec in enumerate(records, start=1):
            if not isinstance(rec, dict):
                malformed_records.append({
                    "record_index": idx,
                    "reason": "Record is not a valid JSON object/dictionary",
                    "raw_type": type(rec).__name__,
                })
                invalid_record_indices.add(idx)
                continue

            # 1. Primary Identity Validation
            comm_raw = rec.get("commodity")
            url_raw = rec.get("article_url") or rec.get("url")
            title_raw = rec.get("article_title") or rec.get("title")

            missing_fields = []
            if comm_raw is None:
                missing_fields.append("commodity")
            if url_raw is None:
                missing_fields.append("article_url")
            if title_raw is None:
                missing_fields.append("article_title")

            if missing_fields:
                malformed_records.append({
                    "record_index": idx,
                    "reason": f"Missing required primary fields: {', '.join(missing_fields)}",
                    "present_keys": list(rec.keys()),
                })
                invalid_record_indices.add(idx)

            # Commodity evaluation
            comm = str(comm_raw or "").strip().lower()
            if comm in self.config.commodities:
                commodity_counts[comm] += 1
            else:
                unsupported_commodities[comm or "missing"] += 1
                invalid_record_indices.add(idx)

            # URL validation
            canonical_url = canonicalize_url(url_raw)
            if not canonical_url:
                missing_urls += 1
                invalid_record_indices.add(idx)
            elif not _is_valid_url(canonical_url):
                invalid_urls.append({
                    "record_index": idx,
                    "commodity": comm,
                    "url": canonical_url,
                    "reason": "URL lacks valid HTTP/HTTPS scheme or netloc structure",
                })
                invalid_record_indices.add(idx)

            # Duplicate identity tracking: composite (commodity, canonical_url)
            if comm and canonical_url:
                key = (comm, canonical_url)
                if key in seen_identities:
                    duplicate_identities.append({
                        "record_index": idx,
                        "commodity": comm,
                        "article_url": canonical_url,
                        "title": str(title_raw or "")[:80],
                    })
                    invalid_record_indices.add(idx)
                else:
                    seen_identities.add(key)

            # 2. Publication Timestamp Validation
            pub_ts_raw = rec.get("article_publication_timestamp") or rec.get("publication_timestamp")
            if not pub_ts_raw:
                missing_timestamps += 1
                invalid_record_indices.add(idx)
            else:
                parsed_ts = parse_timestamp_utc(pub_ts_raw)
                if parsed_ts is None:
                    unparseable_timestamps.append({
                        "record_index": idx,
                        "raw_timestamp": pub_ts_raw,
                        "reason": "Timestamp could not be parsed into valid UTC datetime",
                    })
                    invalid_record_indices.add(idx)
                else:
                    valid_timestamps += 1
                    if parsed_ts > now_utc:
                        future_timestamps.append({
                            "record_index": idx,
                            "timestamp": parsed_ts.isoformat(),
                            "cutoff": cutoff_iso,
                            "lead_seconds": (parsed_ts - now_utc).total_seconds(),
                        })
                        invalid_record_indices.add(idx)

            # 3. Relevance Validation
            rel_class = rec.get("relevance_class")
            rel_score = rec.get("relevance_score")

            if rel_class is not None:
                rel_class_str = str(rel_class).strip().lower()
                relevance_distribution[rel_class_str] += 1
                if rel_class_str not in VALID_RELEVANCE_CLASSES:
                    unsupported_relevance_classes[rel_class_str] += 1
                    invalid_record_indices.add(idx)
            else:
                relevance_distribution["missing"] += 1
                invalid_record_indices.add(idx)

            if not _is_valid_confidence(rel_score):
                invalid_relevance_scores.append({
                    "record_index": idx,
                    "raw_score": rel_score,
                    "reason": "Relevance score is missing, non-numeric, or outside [0.0, 1.0]",
                })
                invalid_record_indices.add(idx)

            # 4. Sentiment Validation
            sent_label = rec.get("sentiment_label")
            sent_score = rec.get("sentiment_score")
            sent_status = rec.get("sentiment_status") or ("missing" if sent_label is None else "success")
            sent_status_clean = str(sent_status).lower()
            sentiment_status_distribution[sent_status_clean] += 1

            if sent_status_clean == "failed":
                failed_sentiment_count += 1
                invalid_record_indices.add(idx)

            if sent_label is not None:
                sent_label_str = str(sent_label).strip().lower()
                sentiment_distribution[sent_label_str] += 1
                if sent_label_str not in VALID_SENTIMENT_LABELS:
                    unsupported_sentiment_labels[sent_label_str] += 1
                    invalid_record_indices.add(idx)
                elif sent_status_clean == "success":
                    successful_sentiment_count += 1

                # If sentiment label is present, sentiment score must be valid confidence
                if not _is_valid_confidence(sent_score):
                    invalid_sentiment_scores.append({
                        "record_index": idx,
                        "raw_score": sent_score,
                        "reason": "Sentiment score is missing, non-numeric, or outside [0.0, 1.0]",
                    })
                    invalid_record_indices.add(idx)
            else:
                sentiment_distribution["missing"] += 1
                missing_sentiment_count += 1

            # 5. Supply-Chain Event Extraction Validation
            ev_type = rec.get("event_type")
            ev_conf = rec.get("event_confidence")
            ev_impact = rec.get("event_impact_direction")
            ev_status = rec.get("extraction_status") or ("missing" if ev_type is None else "success")
            ev_status_clean = str(ev_status).lower()
            extraction_status_distribution[ev_status_clean] += 1

            if ev_status_clean == "failed":
                failed_event_extraction_count += 1
                invalid_record_indices.add(idx)

            if ev_type is not None:
                ev_type_str = str(ev_type).strip().lower()
                event_type_distribution[ev_type_str] += 1
                if ev_type_str not in VALID_EVENT_TYPES:
                    unsupported_event_types[ev_type_str] += 1
                    invalid_record_indices.add(idx)

                if ev_status_clean == "success":
                    successful_event_extraction_count += 1
                    if ev_type_str == "other":
                        no_event_rule_match_count += 1
            else:
                event_type_distribution["missing"] += 1
                missing_event_extraction_count += 1

            if ev_status_clean == "missing" and ev_type is not None:
                missing_event_extraction_count += 1

            if ev_impact is not None:
                ev_impact_str = str(ev_impact).strip().lower()
                impact_direction_distribution[ev_impact_str] += 1
                if ev_impact_str not in VALID_IMPACT_DIRECTIONS:
                    unsupported_impact_directions[ev_impact_str] += 1
                    invalid_record_indices.add(idx)
            else:
                impact_direction_distribution["missing"] += 1

            if not _is_valid_confidence(ev_conf):
                invalid_event_confidences.append({
                    "record_index": idx,
                    "raw_confidence": ev_conf,
                    "reason": "Event confidence is missing, non-numeric, or outside [0.0, 1.0]",
                })
                invalid_record_indices.add(idx)

            # 6. Feature Completeness Status Validation
            feat_status = rec.get("feature_status")
            if feat_status is not None:
                feat_status_str = str(feat_status).strip().lower()
                feature_status_distribution[feat_status_str] += 1
                if feat_status_str not in VALID_FEATURE_STATUSES:
                    unsupported_feature_statuses[feat_status_str] += 1
                    invalid_record_indices.add(idx)
            else:
                feature_status_distribution["missing"] += 1

        # Summary calculations: valid records are total minus unique records with at least one defect
        invalid_records_count = len(invalid_record_indices)
        valid_records_count = max(0, total_records - invalid_records_count)
        valid_ratio = round(valid_records_count / total_records, 4) if total_records > 0 else 0.0

        def _to_percentage_dict(counter: Counter, total: int) -> Dict[str, Dict[str, Any]]:
            res = {}
            for k, count in sorted(counter.items()):
                pct = round((count / total) * 100.0, 2) if total > 0 else 0.0
                res[k] = {"count": count, "percentage": pct}
            return res

        report = {
            "evaluation_metadata": {
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluation_cutoff_applied": cutoff_iso,
                "total_records_evaluated": total_records,
                "valid_records_count": valid_records_count,
                "invalid_records_count": invalid_records_count,
                "data_quality_pass_rate": valid_ratio,
            },
            "volume_and_commodities": {
                "total_records": total_records,
                "counts_by_commodity": commodity_counts,
                "unsupported_commodity_counts": dict(unsupported_commodities),
            },
            "identity_and_integrity": {
                "unique_identities_count": len(seen_identities),
                "duplicate_identities_count": len(duplicate_identities),
                "duplicate_identities_samples": duplicate_identities[:10],
                "malformed_records_count": len(malformed_records),
                "malformed_records_samples": malformed_records[:10],
            },
            "url_validity": {
                "missing_urls_count": missing_urls,
                "invalid_urls_count": len(invalid_urls),
                "invalid_urls_samples": invalid_urls[:10],
            },
            "temporal_validity": {
                "valid_timestamps_count": valid_timestamps,
                "missing_timestamps_count": missing_timestamps,
                "unparseable_timestamps_count": len(unparseable_timestamps),
                "unparseable_timestamps_samples": unparseable_timestamps[:10],
                "future_timestamps_count": len(future_timestamps),
                "future_timestamps_samples": future_timestamps[:10],
                "temporal_availability_note": (
                    "Publication timestamps indicate claimed article publication dates, but do not establish "
                    "the exact point-in-time when the article became ingested or indexed by GDELT."
                ),
            },
            "confidence_value_validity": {
                "invalid_relevance_scores_count": len(invalid_relevance_scores),
                "invalid_relevance_scores_samples": invalid_relevance_scores[:10],
                "invalid_sentiment_scores_count": len(invalid_sentiment_scores),
                "invalid_sentiment_scores_samples": invalid_sentiment_scores[:10],
                "invalid_event_confidences_count": len(invalid_event_confidences),
                "invalid_event_confidences_samples": invalid_event_confidences[:10],
            },
            "distributions": {
                "relevance_classes": _to_percentage_dict(relevance_distribution, total_records),
                "sentiment_labels": _to_percentage_dict(sentiment_distribution, total_records),
                "sentiment_statuses": _to_percentage_dict(sentiment_status_distribution, total_records),
                "event_types": _to_percentage_dict(event_type_distribution, total_records),
                "impact_directions": _to_percentage_dict(impact_direction_distribution, total_records),
                "extraction_statuses": _to_percentage_dict(extraction_status_distribution, total_records),
                "feature_statuses": _to_percentage_dict(feature_status_distribution, total_records),
            },
            "missingness_and_coverage": {
                "successful_sentiment_count": successful_sentiment_count,
                "missing_sentiment_count": missing_sentiment_count,
                "missing_sentiment_rate": round(missing_sentiment_count / total_records, 4) if total_records > 0 else 0.0,
                "failed_sentiment_count": failed_sentiment_count,
                "successful_event_extraction_count": successful_event_extraction_count,
                "missing_event_extraction_count": missing_event_extraction_count,
                "missing_event_extraction_rate": round(missing_event_extraction_count / total_records, 4) if total_records > 0 else 0.0,
                "failed_event_extraction_count": failed_event_extraction_count,
                "no_event_rule_match_count": no_event_rule_match_count,
                "no_event_rule_match_rate": round(no_event_rule_match_count / total_records, 4) if total_records > 0 else 0.0,
                "genuine_absence_of_events_count": no_event_rule_match_count,  # Backward compatible alias
                "genuine_absence_of_events_rate": round(no_event_rule_match_count / total_records, 4) if total_records > 0 else 0.0,
                "epistemic_note": (
                    "no_event_rule_match_count (aliased to genuine_absence_of_events_count for backwards compatibility) "
                    "measures articles where rule-based extraction executed successfully but matched no operational "
                    "event keywords (event_type='other'). Without independent human annotation, this represents "
                    "rule non-detection rather than verified empirical absence of an event."
                ),
            },
            "taxonomy_violations": {
                "unsupported_commodities": dict(unsupported_commodities),
                "unsupported_relevance_classes": dict(unsupported_relevance_classes),
                "unsupported_sentiment_labels": dict(unsupported_sentiment_labels),
                "unsupported_event_types": dict(unsupported_event_types),
                "unsupported_impact_directions": dict(unsupported_impact_directions),
                "unsupported_feature_statuses": dict(unsupported_feature_statuses),
            },
        }

        return report


def validate_article_features(
    filepath: Path,
    output_report_path: Optional[Path] = None,
    evaluation_cutoff: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load a JSONL file, validate records, and optionally save report.
    """
    records = load_jsonl_records(filepath)
    validator = DataQualityValidator()
    report = validator.validate(records, evaluation_cutoff=evaluation_cutoff)

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved data quality report to %s", output_report_path)

    return report
