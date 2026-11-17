"""
Timestamp-safe feature aggregation module for commodity news intelligence.

Aggregates article-level NLP features (sentiment, relevance, supply-chain events)
over strictly backward-looking time windows to prevent data leakage in price
forecasting models.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def parse_timestamp_utc(raw_ts: Optional[Any]) -> Optional[datetime]:
    """
    Safely parse any timestamp string or datetime object into a timezone-aware
    UTC datetime.

    Timezone contract and assumptions:
    1. GDELT DOC 2.0 source timestamps are natively UTC ('YYYYMMDDTHHMMSSZ').
    2. ISO-8601 strings with explicit UTC or offset designators ('Z', '+HH:MM', '-HH:MM')
       are parsed and normalized to UTC.
    3. Timezone-naive datetime strings (e.g. '2026-10-20 12:45:00') lack explicit offset
       information and are assumed to represent UTC by convention. If local timestamps
       are ever ingested from other sources, they must be converted to UTC prior to ingestion.
    4. Date-only strings (e.g. '2026-10-20') are anchored to midnight UTC ('00:00:00Z').
    5. Missing, empty, or unparseable inputs return None and are tracked as unanchored
       records excluded from time-bounded rolling windows.
    """
    if raw_ts is None:
        return None

    if isinstance(raw_ts, datetime):
        if raw_ts.tzinfo is None:
            return raw_ts.replace(tzinfo=timezone.utc)
        return raw_ts.astimezone(timezone.utc)

    if not isinstance(raw_ts, str):
        return None

    cleaned = raw_ts.strip()
    if not cleaned:
        return None

    # Common ISO-8601 UTC format: '2026-10-20T12:45:00Z'
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Alternative ISO-8601 with offset or fractional seconds: '2026-10-20T12:45:00+00:00'
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    # Alternative GDELT compact format: '20261020T124500Z'
    try:
        dt = datetime.strptime(cleaned, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Alternative date only: '2026-10-20'
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Alternative datetime without T: '2026-10-20 12:45:00'
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    logger.debug("Could not parse timestamp string: '%s'", raw_ts)
    return None


def format_iso_utc(dt: datetime) -> str:
    """Format a UTC datetime to standard ISO 8601 UTC string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class TimestampFeatureAggregator:
    """
    Computes backward-looking rolling window aggregations of NLP signals
    with strict cutoff boundary enforcement to prevent future information leakage.
    """

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold

    def aggregate(
        self,
        article_features: List[Dict[str, Any]],
        commodity: str,
        cutoff_timestamp: Any,
        window_days: int,
        cutoff_source: str = "explicit",
    ) -> Dict[str, Any]:
        """
        Aggregate article features for a specific commodity over a backward-looking window.

        Window boundaries:
            window_end = cutoff_timestamp
            window_start = cutoff_timestamp - window_days

        Article inclusion rule:
            window_start <= article_publication_timestamp <= window_end

        Leakage prevention rule:
            Articles with article_publication_timestamp > cutoff_timestamp are strictly excluded.
            Articles with missing or unparseable timestamps are excluded from time-anchored metrics.

        Score Semantics & Definitions:
            - sentiment_confidence_mean: Average model confidence (softmax probability of winning class)
              across articles with sentiment. This measures model certainty, NOT signed sentiment strength.
            - net_sentiment_polarity: Fraction (positive_count - negative_count) / article_count in [-1, +1].
            - confidence_weighted_sentiment_polarity: Signed sum of model confidence (+conf for positive,
              -conf for negative, 0 for neutral) divided by article_count.
            - relevant_net_sentiment_polarity: Polarity computed strictly over relevant articles
              (directly_relevant and potentially_relevant), filtering out irrelevant noise.

        Args:
            article_features: List of standardized article feature dictionaries.
            commodity: Target commodity identifier (e.g. 'copper', 'gold', 'crude oil').
            cutoff_timestamp: Prediction point-in-time cutoff (datetime or ISO 8601 string).
            window_days: Number of days to look backward (e.g. 1 or 5).
            cutoff_source: 'explicit' when set by user/backtester, or 'default_latest_article' when auto-resolved.

        Returns:
            Dictionary containing aggregated NLP features for the window.
        """
        cutoff_dt = parse_timestamp_utc(cutoff_timestamp)
        if cutoff_dt is None:
            raise ValueError(f"Invalid or missing cutoff_timestamp: {cutoff_timestamp}")

        if window_days <= 0:
            raise ValueError(f"window_days must be positive, got {window_days}")

        window_start_dt = cutoff_dt - timedelta(days=window_days)
        norm_comm = commodity.strip().lower()

        # Audit counters
        future_filtered_count = 0
        unanchored_excluded_count = 0
        total_evaluated = 0

        # Unique articles in window (deduplicated by canonical article_url)
        included_articles: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        for article in article_features:
            art_comm = str(article.get("commodity", "")).strip().lower()
            if art_comm != norm_comm:
                continue

            total_evaluated += 1
            raw_ts = article.get("article_publication_timestamp")
            art_dt = parse_timestamp_utc(raw_ts)

            if art_dt is None:
                unanchored_excluded_count += 1
                continue

            # Strict future article check (leakage prevention)
            if art_dt > cutoff_dt:
                future_filtered_count += 1
                continue

            # Window start check
            if art_dt < window_start_dt:
                continue

            # In window: deduplicate by canonical URL
            url = str(article.get("article_url", "")).strip()
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

            included_articles.append(article)

        # Compute aggregated statistics
        n_articles = len(included_articles)

        # Relevance aggregations
        rel_counts = {
            "directly_relevant": 0,
            "potentially_relevant": 0,
            "irrelevant": 0,
            "uncertain": 0,
        }
        rel_scores: List[float] = []

        # All-article sentiment aggregations
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "missing": 0,
        }
        sentiment_confidences: List[float] = []
        signed_sentiment_scores: List[float] = []

        # Relevant-only sentiment aggregations (excludes irrelevant and uncertain noise)
        relevant_sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "missing": 0,
        }
        relevant_signed_scores: List[float] = []

        # Event type aggregations
        event_type_counts = {
            "supply_disruption": 0,
            "demand_change": 0,
            "geopolitical_risk": 0,
            "inventory_change": 0,
            "policy_action": 0,
            "production_change": 0,
            "other": 0,
        }

        # Event impact direction aggregations
        event_impact_counts = {
            "potentially_bullish": 0,
            "potentially_bearish": 0,
            "mixed": 0,
            "unknown": 0,
        }
        high_conf_event_count = 0
        relevant_high_conf_event_count = 0

        for art in included_articles:
            # Relevance
            r_class = art.get("relevance_class", "uncertain")
            rel_counts[r_class] = rel_counts.get(r_class, 0) + 1
            if "relevance_score" in art and art["relevance_score"] is not None:
                rel_scores.append(float(art["relevance_score"]))

            is_relevant_article = r_class in ("directly_relevant", "potentially_relevant")

            # Sentiment
            s_label = art.get("sentiment_label")
            raw_s_score = art.get("sentiment_score")
            s_conf = float(raw_s_score) if raw_s_score is not None else 0.0

            if s_label in ("positive", "negative", "neutral"):
                sentiment_counts[s_label] += 1
                if raw_s_score is not None:
                    sentiment_confidences.append(s_conf)
                # Signed contribution: +conf for positive, -conf for negative, 0 for neutral
                if s_label == "positive":
                    signed_sentiment_scores.append(s_conf)
                elif s_label == "negative":
                    signed_sentiment_scores.append(-s_conf)
                else:
                    signed_sentiment_scores.append(0.0)

                # Track separately for relevant articles
                if is_relevant_article:
                    relevant_sentiment_counts[s_label] += 1
                    if s_label == "positive":
                        relevant_signed_scores.append(s_conf)
                    elif s_label == "negative":
                        relevant_signed_scores.append(-s_conf)
                    else:
                        relevant_signed_scores.append(0.0)
            else:
                sentiment_counts["missing"] += 1
                signed_sentiment_scores.append(0.0)
                if is_relevant_article:
                    relevant_sentiment_counts["missing"] += 1
                    relevant_signed_scores.append(0.0)

            # Events
            e_type = art.get("event_type", "other")
            event_type_counts[e_type] = event_type_counts.get(e_type, 0) + 1

            impact_dir = art.get("event_impact_direction", "unknown")
            event_impact_counts[impact_dir] = event_impact_counts.get(impact_dir, 0) + 1

            conf = float(art.get("event_confidence", 0.0))
            if conf >= self.confidence_threshold and e_type != "other":
                high_conf_event_count += 1
                if is_relevant_article:
                    relevant_high_conf_event_count += 1

        # Mean computations
        rel_score_mean = round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None

        # Model confidence mean (softmax probability of winning class, NOT directional strength)
        sentiment_confidence_mean = (
            round(sum(sentiment_confidences) / len(sentiment_confidences), 4)
            if sentiment_confidences
            else None
        )

        # Net sentiment polarity: (pos - neg) / article_count in [-1.0, +1.0]
        net_sentiment_polarity = (
            round((sentiment_counts["positive"] - sentiment_counts["negative"]) / n_articles, 4)
            if n_articles > 0
            else 0.0
        )

        # Confidence-weighted polarity across all window articles: sum(signed_scores) / article_count
        confidence_weighted_sentiment_polarity = (
            round(sum(signed_sentiment_scores) / n_articles, 4)
            if n_articles > 0
            else 0.0
        )

        # Relevant article counts and clean metrics (isolated from sports/slang noise)
        relevant_article_count = (
            rel_counts["directly_relevant"] + rel_counts["potentially_relevant"]
        )

        relevant_net_sentiment_polarity = (
            round(
                (relevant_sentiment_counts["positive"] - relevant_sentiment_counts["negative"])
                / relevant_article_count,
                4,
            )
            if relevant_article_count > 0
            else 0.0
        )

        relevant_confidence_weighted_polarity = (
            round(sum(relevant_signed_scores) / relevant_article_count, 4)
            if relevant_article_count > 0
            else 0.0
        )

        return {
            "commodity": norm_comm,
            "cutoff_timestamp": format_iso_utc(cutoff_dt),
            "cutoff_source": cutoff_source,
            "window_days": window_days,
            "window_start_timestamp": format_iso_utc(window_start_dt),
            "window_end_timestamp": format_iso_utc(cutoff_dt),
            "article_count": n_articles,
            "relevant_article_count": relevant_article_count,
            "directly_relevant_count": rel_counts["directly_relevant"],
            "potentially_relevant_count": rel_counts["potentially_relevant"],
            "irrelevant_count": rel_counts["irrelevant"],
            "uncertain_count": rel_counts["uncertain"],
            "relevance_score_mean": rel_score_mean,
            # All-article sentiment metrics
            "sentiment_positive_count": sentiment_counts["positive"],
            "sentiment_negative_count": sentiment_counts["negative"],
            "sentiment_neutral_count": sentiment_counts["neutral"],
            "sentiment_missing_count": sentiment_counts["missing"],
            "sentiment_confidence_mean": sentiment_confidence_mean,
            "sentiment_score_mean": sentiment_confidence_mean,  # Kept as identical alias for backward compatibility
            "net_sentiment_polarity": net_sentiment_polarity,
            "confidence_weighted_sentiment_polarity": confidence_weighted_sentiment_polarity,
            # Relevant-only sentiment metrics (isolated from irrelevant sports/slang)
            "relevant_sentiment_positive_count": relevant_sentiment_counts["positive"],
            "relevant_sentiment_negative_count": relevant_sentiment_counts["negative"],
            "relevant_sentiment_neutral_count": relevant_sentiment_counts["neutral"],
            "relevant_net_sentiment_polarity": relevant_net_sentiment_polarity,
            "relevant_confidence_weighted_polarity": relevant_confidence_weighted_polarity,
            # Supply-chain event metrics
            "event_type_counts": event_type_counts,
            "event_impact_counts": event_impact_counts,
            "high_confidence_event_count": high_conf_event_count,
            "relevant_high_confidence_event_count": relevant_high_conf_event_count,
            # Leakage & boundary auditing
            "future_articles_filtered": future_filtered_count,
            "unanchored_articles_excluded": unanchored_excluded_count,
        }

    def aggregate_windows(
        self,
        article_features: List[Dict[str, Any]],
        commodities: Optional[List[str]] = None,
        cutoff_timestamp: Optional[Any] = None,
        window_days_list: Optional[List[int]] = None,
        cutoff_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute rolling window aggregations across multiple commodities and window horizons.
        """
        windows = window_days_list or [1, 5]
        target_commodities = (
            [c.strip().lower() for c in commodities]
            if commodities
            else ["copper", "gold", "crude oil"]
        )

        resolved_source = cutoff_source or ("explicit" if cutoff_timestamp is not None else "default_latest_article")

        # If cutoff_timestamp is not provided, use the latest valid publication timestamp in articles
        if cutoff_timestamp is None:
            latest_dt: Optional[datetime] = None
            for art in article_features:
                ts = parse_timestamp_utc(art.get("article_publication_timestamp"))
                if ts:
                    if latest_dt is None or ts > latest_dt:
                        latest_dt = ts
            cutoff_dt = latest_dt or datetime.now(timezone.utc)
        else:
            cutoff_dt = parse_timestamp_utc(cutoff_timestamp)
            if cutoff_dt is None:
                raise ValueError(f"Invalid cutoff_timestamp: {cutoff_timestamp}")

        results: List[Dict[str, Any]] = []
        for comm in target_commodities:
            for w_days in windows:
                agg = self.aggregate(
                    article_features=article_features,
                    commodity=comm,
                    cutoff_timestamp=cutoff_dt,
                    window_days=w_days,
                    cutoff_source=resolved_source,
                )
                results.append(agg)
        return results
