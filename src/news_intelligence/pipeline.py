"""
News Ingestion Pipeline orchestrator for AI-powered Commodity Price Prediction.
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .gdelt_client import GDELTClient, GDELTClientError
from .normalizer import normalize_article
from .storage import append_records, load_existing_keys

logger = logging.getLogger(__name__)


class NewsIngestionPipeline:
    """
    Orchestrates the ingestion, normalization, deduplication, and storage of commodity news.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        client: Optional[GDELTClient] = None,
    ):
        self.config = config or PipelineConfig()
        self.client = client or GDELTClient(config=self.config)

    def run(
        self,
        commodities: Optional[List[str]] = None,
        max_records: Optional[int] = None,
        timespan: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute the news ingestion pipeline for specified commodities.

        Args:
            commodities: List of commodities to fetch (defaults to config.commodities).
            max_records: Maximum records to request per commodity.
            timespan: Timespan window (e.g., '24h', '7d').
            output_path: Destination .jsonl path.

        Returns:
            Dictionary summary of the pipeline execution run.
        """
        target_commodities = commodities or self.config.commodities
        target_output = Path(output_path or self.config.output_path)
        records_per_query = max_records or self.config.max_records
        time_window = timespan or self.config.timespan

        logger.info(
            "Starting News Ingestion Pipeline for %d commodities: %s",
            len(target_commodities),
            target_commodities,
        )

        # Pre-load existing (commodity, url) keys to ensure deduplication
        existing_keys = load_existing_keys(target_output)

        summary: Dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_commodities": target_commodities,
            "output_path": str(target_output),
            "total_fetched": 0,
            "total_saved": 0,
            "total_duplicates": 0,
            "commodity_stats": {},
            "errors": {},
            "success": True,
        }

        fetch_batch_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for commodity in target_commodities:
            norm_commodity = commodity.strip().lower()
            query = self.config.commodity_queries.get(norm_commodity, norm_commodity)

            logger.info("Processing commodity '%s' with query '%s'", norm_commodity, query)

            try:
                raw_articles = self.client.fetch_articles(
                    query=query,
                    max_records=records_per_query,
                    timespan=time_window,
                )
            except GDELTClientError as err:
                logger.error(
                    "Failed to fetch articles for commodity '%s': %s",
                    norm_commodity,
                    err,
                )
                summary["errors"][norm_commodity] = str(err)
                summary["commodity_stats"][norm_commodity] = {
                    "fetched": 0,
                    "saved": 0,
                    "duplicates": 0,
                    "error": str(err),
                }
                # Continue safely to the next commodity (partial failure handling)
                continue

            # Normalize raw articles
            normalized_records: List[Dict[str, Any]] = []
            for raw_art in raw_articles:
                norm_record = normalize_article(
                    raw_art,
                    commodity=norm_commodity,
                    fetch_timestamp=fetch_batch_timestamp,
                )
                if norm_record:
                    normalized_records.append(norm_record)

            # Deduplicate and save
            saved_count, dup_count = append_records(
                filepath=target_output,
                records=normalized_records,
                existing_keys=existing_keys,
            )

            summary["total_fetched"] += len(raw_articles)
            summary["total_saved"] += saved_count
            summary["total_duplicates"] += dup_count
            summary["commodity_stats"][norm_commodity] = {
                "fetched": len(raw_articles),
                "saved": saved_count,
                "duplicates": dup_count,
            }

        summary["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if summary["errors"] and summary["total_saved"] == 0:
            summary["success"] = False

        logger.info(
            "Pipeline completed: %d fetched, %d saved, %d duplicates skipped across %s",
            summary["total_fetched"],
            summary["total_saved"],
            summary["total_duplicates"],
            target_output,
        )

        return summary


def main() -> None:
    """Command-line entrypoint for the news ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest commodity news from GDELT DOC 2.0 API into JSONL."
    )
    parser.add_argument(
        "--commodities",
        type=str,
        default="copper,gold,crude oil",
        help="Comma-separated list of commodities to fetch (default: 'copper,gold,crude oil')",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=25,
        help="Maximum records to request per commodity (default: 25)",
    )
    parser.add_argument(
        "--timespan",
        type=str,
        default="7d",
        help="GDELT timespan window (e.g. '24h', '7d', '1m', default: '7d')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/commodity_news.jsonl",
        help="Destination JSONL output path (default: data/raw/commodity_news.jsonl)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=5.5,
        help="Delay in seconds between GDELT API calls (default: 5.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    commodities = [c.strip() for c in args.commodities.split(",") if c.strip()]
    config = PipelineConfig(
        commodities=commodities,
        max_records=args.max_records,
        timespan=args.timespan,
        output_path=Path(args.output),
        rate_limit_delay_seconds=args.rate_limit_delay,
    )

    pipeline = NewsIngestionPipeline(config=config)
    result = pipeline.run(
        commodities=commodities,
        max_records=args.max_records,
        timespan=args.timespan,
        output_path=Path(args.output),
    )

    print("\n--- Pipeline Execution Summary ---")
    print(f"Status:            {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Output File:       {result['output_path']}")
    print(f"Total Fetched:     {result['total_fetched']}")
    print(f"Total Saved:       {result['total_saved']}")
    print(f"Total Duplicates:  {result['total_duplicates']}")
    for comm, stats in result["commodity_stats"].items():
        if "error" in stats:
            print(f"  [{comm}] ERROR: {stats['error']}")
        else:
            print(f"  [{comm}] Fetched: {stats['fetched']}, Saved: {stats['saved']}, Duplicates: {stats['duplicates']}")

    if result["errors"]:
        print("\nWarnings / Errors encountered:")
        for comm, err_msg in result["errors"].items():
            print(f"  - {comm}: {err_msg}")


if __name__ == "__main__":
    main()
