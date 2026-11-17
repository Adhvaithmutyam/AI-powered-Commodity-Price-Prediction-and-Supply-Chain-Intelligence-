"""
Human Annotation Workflow and Evaluation Sample Generation.

Generates reproducible, stratified, unbiased annotation templates from ingested
news or feature datasets. Eliminates model prediction leakage to ensure objective
human ground-truth evaluation.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from src.features.builder import canonicalize_url, load_jsonl_records
from src.evaluation.config import (
    EvaluationConfig,
    VALID_COMMODITIES,
    VALID_RELEVANCE_CLASSES,
    VALID_SENTIMENT_LABELS,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comprehensive Annotation Guidelines
# ---------------------------------------------------------------------------
ANNOTATION_GUIDELINES: str = """# Commodity News Intelligence Annotation Guidelines

## Purpose
This document provides objective criteria for human annotators creating ground-truth
evaluation datasets for commodity news relevance, financial sentiment, and supply-chain events.

CRITICAL PRINCIPLE:
Do not consult automated model outputs or attempt to reverse-engineer rule patterns.
Base your annotations strictly on the evidence available in the article headline and metadata.

---

## 1. News Relevance (`true_relevance_class`)
Evaluate how directly the article informs commodity supply, demand, or price dynamics.

| Class | Definition & Criteria | Examples |
| :--- | :--- | :--- |
| `directly_relevant` | Directly discusses commercial mining, refining, production, physical trading, inventories, tariffs, or supply disruptions for the specific commodity. | • "Escondida copper mine workers commence indefinite strike"<br/>• "OPEC+ agrees to extend 2.2M bpd crude oil output cut"<br/>• "Gold central bank purchases hit record quarterly high" |
| `potentially_relevant` | Macroeconomic, monetary, cross-asset, or trade policy developments that strongly influence commodity prices or industrial demand channels. | • "Federal Reserve signals 50 bps interest rate reduction"<br/>• "US CPI inflation accelerates to 3.8% annually"<br/>• "China manufacturing PMI contracts unexpectedly to 49.1" |
| `irrelevant` | Unrelated sports competitions, figurative speech, slang, entertainment, unrelated corporate names, or false positives. | • "Sprinter wins Olympic gold medal in Paris"<br/>• "CEO made a crude joke during the quarterly earnings call"<br/>• "Copper Mountain ski resort announces opening date" |
| `uncertain` | Missing, blank, or completely ambiguous headline where subject matter cannot be determined without full body text. | • "Breaking news update for today"<br/>• "Market overview and perspectives"<br/>• "Exclusive interview: what comes next" |

---

## 2. Financial Sentiment (`true_sentiment_label`)
Evaluate the financial sentiment polarity of the headline from the perspective of market participants.

| Label | Definition | Examples |
| :--- | :--- | :--- |
| `positive` | Positive financial tone, price gains, revenue growth, or favorable market developments. | • "Copper futures rally to multi-year high on strong industrial demand"<br/>• "Oil majors report record quarterly profits" |
| `negative` | Negative financial tone, price declines, losses, operational distress, or severe disruptions. | • "Gold prices slide as Treasury yields climb"<br/>• "Refinery explosion shuts down key fuel processing units" |
| `neutral` | Factual, balanced, administrative, or non-directional market reporting. | • "EIA scheduled to release weekly petroleum inventory report on Wednesday"<br/>• "Mining conference opens in Santiago today" |

---

## 3. Supply-Chain Event Type (`true_event_type`)
Classify physical and operational supply-chain events impacting the physical commodity.

| Event Type | Scope |
| :--- | :--- |
| `supply_disruption` | Mine closures, strikes, pipeline leaks, refinery explosions, shipping blockades, transit bottlenecks. |
| `demand_change` | Major shifts in industrial consumption, EV manufacturing changes, consumer demand, strategic reserve buying. |
| `geopolitical_risk` | War, military blockades, rebel attacks, regional trade sanctions, sanctions enforcement. |
| `inventory_change` | Commercial stockpile draws or builds reported by LME, COMEX, SHFE, or EIA/API weekly reports. |
| `policy_action` | Export taxes, import tariffs, environmental regulation, mining royalties, strategic stockpile releases. |
| `production_change` | OPEC+ quota increases/cuts, mining expansion projects, smelter capacity curtailments. |
| `other` | General news, macroeconomic reports, or articles with no specific physical supply-chain event. |

---

## 4. Expected Impact Direction (`true_event_impact_direction`)
The theoretical directional effect of the physical event on physical commodity price (supply/demand balance):
- `potentially_bullish`: Supply reductions, inventory draws, strong demand surges.
- `potentially_bearish`: Supply increases, quota hikes, inventory builds, demand collapse.
- `mixed`: Conflicting signals (e.g. higher production offset by surging demand).
- `unknown`: No clear directional effect or non-operational event.
"""


def _generate_deterministic_id(commodity: str, url: str) -> str:
    """Generate a stable, unique 12-character article identifier using canonicalized URL."""
    c_url = canonicalize_url(url) or str(url or "").strip()
    raw = f"{commodity.strip().lower()}::{c_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class AnnotationSampler:
    """
    Constructs deterministic, stratified evaluation sample datasets for human annotation.
    Guarantees no leakage of model predictions into labeling templates.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    def sample(
        self,
        records: List[Dict[str, Any]],
        sample_size: int = 50,
        seed: int = 42,
        stratify_by_commodity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract a deduplicated, stratified, bias-free sample for human labeling.

        Args:
            records: List of article or feature records.
            sample_size: Total number of records to sample.
            seed: Fixed random seed for complete reproducibility.
            stratify_by_commodity: Whether to balance sample across target commodities.

        Returns:
            List of blank annotation templates containing only article context.
        """
        if not records or sample_size <= 0:
            return []

        # 1. Deduplicate by (commodity, canonical_url) and filter to supported commodities
        seen: Set[Tuple[str, str]] = set()
        by_commodity: Dict[str, List[Dict[str, Any]]] = {comm: [] for comm in self.config.commodities}

        for rec in records:
            if not isinstance(rec, dict):
                continue
            comm = str(rec.get("commodity") or "").strip().lower()
            if comm not in self.config.commodities:
                continue

            url_raw = rec.get("article_url") or rec.get("url") or rec.get("evidence_url")
            url = canonicalize_url(url_raw)
            if not url:
                continue

            title = str(rec.get("article_title") or rec.get("title") or rec.get("event_summary") or "").strip()
            if not title:
                continue

            key = (comm, url)
            if key in seen:
                continue
            seen.add(key)

            pub_ts = rec.get("article_publication_timestamp") or rec.get("publication_timestamp")

            clean_item = {
                "article_id": _generate_deterministic_id(comm, url),
                "commodity": comm,
                "article_url": url,
                "article_title": title,
                "article_publication_timestamp": pub_ts,
                "domain": rec.get("domain") or "",
                "language": rec.get("language") or "unknown",
                "source_country": rec.get("source_country"),
                # Blank human ground-truth fields to be populated by annotators
                "true_relevance_class": "",
                "true_sentiment_label": "",
                "true_event_type": "",
                "true_event_impact_direction": "",
                "annotator_id": "",
                "annotator_notes": "",
            }
            by_commodity[comm].append(clean_item)

        rng = random.Random(seed)

        # 2. Perform stratified or uniform deterministic sampling
        sampled_records: List[Dict[str, Any]] = []

        if stratify_by_commodity:
            active_commodities = [c for c in self.config.commodities if by_commodity[c]]
            if not active_commodities:
                return []

            target_per_comm = sample_size // len(active_commodities)
            remainder = sample_size % len(active_commodities)

            # Deterministically shuffle each commodity pool
            for idx, comm in enumerate(active_commodities):
                pool = by_commodity[comm]
                rng.shuffle(pool)
                quota = target_per_comm + (1 if idx < remainder else 0)
                sampled_records.extend(pool[:quota])

            # Re-shuffle combined sample deterministically
            rng.shuffle(sampled_records)
        else:
            flat_pool = [item for pool in by_commodity.values() for item in pool]
            rng.shuffle(flat_pool)
            sampled_records = flat_pool[:sample_size]

        return sampled_records


def generate_annotation_sample(
    input_path: Path,
    output_path: Path,
    sample_size: int = 50,
    seed: int = 42,
    stratify: bool = True,
    include_guidelines: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to load records, sample, and save annotation JSONL template.
    Optionally writes an accompanying GUIDELINES.md file.
    """
    records = load_jsonl_records(input_path)
    sampler = AnnotationSampler()
    sample = sampler.sample(records, sample_size=sample_size, seed=seed, stratify_by_commodity=stratify)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d annotation templates to %s", len(sample), output_path)

    if include_guidelines:
        guide_path = output_path.parent / "ANNOTATION_GUIDELINES.md"
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(ANNOTATION_GUIDELINES)
        logger.info("Saved annotation guidelines to %s", guide_path)

    return sample


def validate_annotation_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Audit an annotated dataset to verify label formatting, completion rate,
    duplicate annotations, and taxonomy compliance prior to downstream evaluation.
    """
    total = len(records)
    labeled_relevance = 0
    labeled_sentiment = 0
    labeled_events = 0
    fully_labeled = 0

    seen_annotations: Set[Tuple[str, str]] = set()
    duplicate_annotations: List[Dict[str, Any]] = []

    relevance_counts: Counter = Counter()
    sentiment_counts: Counter = Counter()
    event_counts: Counter = Counter()

    invalid_relevance: List[Dict[str, Any]] = []
    invalid_sentiment: List[Dict[str, Any]] = []
    invalid_events: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue

        comm = str(rec.get("commodity") or "").strip().lower()
        url = canonicalize_url(rec.get("article_url") or rec.get("url"))
        art_id = rec.get("article_id")

        if comm and url:
            dedup_key = (comm, url)
            if dedup_key in seen_annotations:
                duplicate_annotations.append({
                    "record_index": idx,
                    "commodity": comm,
                    "article_url": url,
                    "article_id": art_id,
                })
            else:
                seen_annotations.add(dedup_key)

        rel = str(rec.get("true_relevance_class") or "").strip().lower()
        sent = str(rec.get("true_sentiment_label") or "").strip().lower()
        ev = str(rec.get("true_event_type") or "").strip().lower()

        rel_valid = False
        sent_valid = False
        ev_valid = False

        if rel:
            labeled_relevance += 1
            relevance_counts[rel] += 1
            if rel not in VALID_RELEVANCE_CLASSES:
                invalid_relevance.append({"record_index": idx, "value": rel})
            else:
                rel_valid = True

        if sent:
            labeled_sentiment += 1
            sentiment_counts[sent] += 1
            if sent not in VALID_SENTIMENT_LABELS:
                invalid_sentiment.append({"record_index": idx, "value": sent})
            else:
                sent_valid = True

        if ev:
            labeled_events += 1
            event_counts[ev] += 1
            if ev not in VALID_EVENT_TYPES:
                invalid_events.append({"record_index": idx, "value": ev})
            else:
                ev_valid = True

        if rel_valid and sent_valid and ev_valid:
            fully_labeled += 1

    return {
        "total_records": total,
        "completion_status": {
            "fully_labeled_records": fully_labeled,
            "fully_labeled_rate": round(fully_labeled / total, 4) if total > 0 else 0.0,
            "labeled_relevance_count": labeled_relevance,
            "labeled_relevance_rate": round(labeled_relevance / total, 4) if total > 0 else 0.0,
            "labeled_sentiment_count": labeled_sentiment,
            "labeled_sentiment_rate": round(labeled_sentiment / total, 4) if total > 0 else 0.0,
            "labeled_events_count": labeled_events,
            "labeled_events_rate": round(labeled_events / total, 4) if total > 0 else 0.0,
        },
        "duplicate_detection": {
            "unique_annotations_count": len(seen_annotations),
            "duplicate_annotations_count": len(duplicate_annotations),
            "duplicate_annotations_samples": duplicate_annotations[:10],
        },
        "label_distributions": {
            "relevance": dict(relevance_counts),
            "sentiment": dict(sentiment_counts),
            "events": dict(event_counts),
        },
        "taxonomy_violations": {
            "invalid_relevance_count": len(invalid_relevance),
            "invalid_relevance_samples": invalid_relevance[:10],
            "invalid_sentiment_count": len(invalid_sentiment),
            "invalid_sentiment_samples": invalid_sentiment[:10],
            "invalid_events_count": len(invalid_events),
            "invalid_events_samples": invalid_events[:10],
        },
    }
