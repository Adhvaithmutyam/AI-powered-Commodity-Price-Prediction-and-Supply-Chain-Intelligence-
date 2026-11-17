"""
NLP Feature Pipeline orchestrator.

Coordinates article-level feature building and timestamp-safe window aggregation
for downstream commodity price prediction.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .builder import ArticleFeatureBuilder, load_jsonl_records
from .aggregator import TimestampFeatureAggregator, parse_timestamp_utc, format_iso_utc
from .config import FeatureConfig

logger = logging.getLogger(__name__)


class NLPFeaturePipeline:
    """
    Orchestrates building article-level NLP features and windowed time aggregations.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self.builder = ArticleFeatureBuilder()
        self.aggregator = TimestampFeatureAggregator(
            confidence_threshold=self.config.event_confidence_threshold
        )

    def run(
        self,
        raw_news_path: Optional[Path] = None,
        sentiment_path: Optional[Path] = None,
        events_path: Optional[Path] = None,
        output_article_path: Optional[Path] = None,
        output_aggregated_path: Optional[Path] = None,
        cutoff_timestamp: Optional[Any] = None,
        window_days_list: Optional[List[int]] = None,
        commodities: Optional[List[str]] = None,
        build_aggregated: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the NLP feature extraction and aggregation pipeline.

        Args:
            raw_news_path: Path to raw commodity news JSONL.
            sentiment_path: Path to sentiment-enriched news JSONL.
            events_path: Path to extracted supply-chain events JSONL.
            output_article_path: Destination path for article-level features JSONL.
            output_aggregated_path: Destination path for aggregated features JSONL.
            cutoff_timestamp: Prediction point-in-time cutoff.
            window_days_list: Rolling window sizes in days (e.g. [1, 5]).
            commodities: Target commodities to aggregate for.
            build_aggregated: Whether to compute windowed aggregations.

        Returns:
            Dictionary summary of the feature pipeline execution run.
        """
        raw_file = Path(raw_news_path or self.config.raw_news_path)
        sent_file = Path(sentiment_path or self.config.sentiment_path)
        event_file = Path(events_path or self.config.events_path)
        out_art = Path(output_article_path or self.config.article_features_path)
        out_agg = Path(output_aggregated_path or self.config.aggregated_features_path)
        target_commodities = commodities or self.config.commodities
        target_windows = window_days_list or self.config.aggregation_window_days

        logger.info("Starting NLP Feature Pipeline...")
        logger.info("Loading source datasets: raw=%s, sentiment=%s, events=%s", raw_file, sent_file, event_file)

        raw_records = load_jsonl_records(raw_file)
        sent_records = load_jsonl_records(sent_file)
        event_records = load_jsonl_records(event_file)

        logger.info(
            "Loaded records: raw=%d, sentiment=%d, events=%d",
            len(raw_records),
            len(sent_records),
            len(event_records),
        )

        # 1. Build article-level features
        article_features = self.builder.build(
            news_records=raw_records,
            sentiment_records=sent_records,
            event_records=event_records,
        )
        logger.info("Built %d article-level NLP feature records", len(article_features))

        # Save article features
        out_art.parent.mkdir(parents=True, exist_ok=True)
        with open(out_art, "w", encoding="utf-8") as f:
            for art in article_features:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")
        logger.info("Saved article-level features to %s", out_art)

        # 2. Window Aggregation (if requested)
        aggregated_features: List[Dict[str, Any]] = []
        resolved_cutoff_str: Optional[str] = None

        if build_aggregated:
            # Determine cutoff
            if cutoff_timestamp is not None:
                parsed_cutoff = parse_timestamp_utc(cutoff_timestamp)
                if parsed_cutoff is None:
                    raise ValueError(f"Invalid cutoff_timestamp provided: {cutoff_timestamp}")
                cutoff_dt = parsed_cutoff
                cutoff_source = "explicit"
            else:
                # Find latest publication timestamp in article features
                latest_dt: Optional[datetime] = None
                for art in article_features:
                    ts = parse_timestamp_utc(art.get("article_publication_timestamp"))
                    if ts:
                        if latest_dt is None or ts > latest_dt:
                            latest_dt = ts
                cutoff_dt = latest_dt or datetime.now(timezone.utc)
                cutoff_source = "default_latest_article"

            resolved_cutoff_str = format_iso_utc(cutoff_dt)
            if cutoff_source == "default_latest_article":
                logger.warning(
                    "No explicit prediction cutoff provided. Defaulting to latest dataset timestamp (%s). "
                    "FOR HISTORICAL BACKTESTING, AN EXPLICIT --cutoff MUST BE SPECIFIED to prevent lookahead leakage.",
                    resolved_cutoff_str,
                )

            aggregated_features = self.aggregator.aggregate_windows(
                article_features=article_features,
                commodities=target_commodities,
                cutoff_timestamp=cutoff_dt,
                window_days_list=target_windows,
                cutoff_source=cutoff_source,
            )

            out_agg.parent.mkdir(parents=True, exist_ok=True)
            with open(out_agg, "w", encoding="utf-8") as f:
                for agg in aggregated_features:
                    f.write(json.dumps(agg, ensure_ascii=False) + "\n")
            logger.info("Saved %d windowed feature records to %s", len(aggregated_features), out_agg)

        summary = {
            "success": True,
            "article_features_count": len(article_features),
            "article_features_path": str(out_art),
            "aggregated_features_count": len(aggregated_features),
            "aggregated_features_path": str(out_agg) if build_aggregated else None,
            "commodities": target_commodities,
            "aggregation_windows": target_windows if build_aggregated else [],
            "cutoff_timestamp": resolved_cutoff_str,
            "cutoff_source": cutoff_source if build_aggregated else None,
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return summary


def main() -> None:
    """Command-line entrypoint for the NLP feature pipeline."""
    # Ensure UTF-8 output on Windows consoles
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Build standardized, timestamp-aware NLP feature datasets for commodity price forecasting."
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="data/raw/commodity_news.jsonl",
        help="Path to raw news JSONL (default: data/raw/commodity_news.jsonl)",
    )
    parser.add_argument(
        "--sentiment-file",
        type=str,
        default="data/processed/commodity_news_sentiment.jsonl",
        help="Path to sentiment JSONL (default: data/processed/commodity_news_sentiment.jsonl)",
    )
    parser.add_argument(
        "--events-file",
        type=str,
        default="data/processed/commodity_events.jsonl",
        help="Path to events JSONL (default: data/processed/commodity_events.jsonl)",
    )
    parser.add_argument(
        "--output-articles",
        type=str,
        default="data/processed/nlp_article_features.jsonl",
        help="Output path for article features (default: data/processed/nlp_article_features.jsonl)",
    )
    parser.add_argument(
        "--output-aggregated",
        type=str,
        default="data/processed/nlp_aggregated_features.jsonl",
        help="Output path for aggregated features (default: data/processed/nlp_aggregated_features.jsonl)",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Explicit prediction cutoff timestamp (ISO 8601 UTC, e.g. '2026-10-20T12:00:00Z')",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="1,5",
        help="Comma-separated aggregation windows in days (default: '1,5')",
    )
    parser.add_argument(
        "--commodities",
        type=str,
        default="copper,gold,crude oil",
        help="Comma-separated commodities list (default: 'copper,gold,crude oil')",
    )
    parser.add_argument(
        "--skip-aggregated",
        action="store_true",
        help="Skip computing windowed aggregations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]
    commodities = [c.strip().lower() for c in args.commodities.split(",") if c.strip()]

    pipeline = NLPFeaturePipeline()
    summary = pipeline.run(
        raw_news_path=Path(args.raw_file),
        sentiment_path=Path(args.sentiment_file),
        events_path=Path(args.events_file),
        output_article_path=Path(args.output_articles),
        output_aggregated_path=Path(args.output_aggregated),
        cutoff_timestamp=args.cutoff,
        window_days_list=windows,
        commodities=commodities,
        build_aggregated=not args.skip_aggregated,
    )

    print("\n=======================================================")
    print(" NLP Feature Pipeline Execution Summary")
    print("=======================================================")
    print(f"Status:                      SUCCESS")
    print(f"Article Features Count:      {summary['article_features_count']}")
    print(f"Article Features File:       {summary['article_features_path']}")
    if summary["aggregated_features_count"]:
        print(f"Aggregated Windows Count:    {summary['aggregated_features_count']}")
        print(f"Aggregated Features File:    {summary['aggregated_features_path']}")
        print(f"Prediction Cutoff Used:      {summary['cutoff_timestamp']}")
        print(f"Aggregation Windows:         {summary['aggregation_windows']} days")
        print(f"Target Commodities:          {summary['commodities']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
