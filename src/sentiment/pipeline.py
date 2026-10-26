"""
Pipeline orchestrator for reading raw commodity news, applying FinBERT sentiment analysis,
and persisting enriched records to data/processed/.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .analyzer import FinBERTSentimentAnalyzer
from .config import SentimentConfig

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    Coordinates reading raw news records, running batched FinBERT sentiment inference,
    and writing enriched records to the processed data directory.
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        analyzer: Optional[FinBERTSentimentAnalyzer] = None,
    ):
        self.config = config or SentimentConfig()
        self.analyzer = analyzer or FinBERTSentimentAnalyzer(config=self.config)

    def load_raw_records(self, input_path: Path) -> List[Dict[str, Any]]:
        """Read and validate raw commodity news records from a JSONL file."""
        records: List[Dict[str, Any]] = []
        if not input_path.exists():
            logger.warning("Input file does not exist: %s", input_path)
            return records

        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSON on line %d of %s: %s", line_no, input_path, e)

        logger.info("Loaded %d raw records from %s", len(records), input_path)
        return records

    def run(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute sentiment enrichment on raw news data.

        Returns summary dictionary of processing statistics.
        """
        src_path = Path(input_path or self.config.input_path)
        dst_path = Path(output_path or self.config.output_path)

        summary: Dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_file": str(src_path),
            "output_file": str(dst_path),
            "model_name": self.config.model_name,
            "total_read": 0,
            "total_enriched": 0,
            "sentiment_counts": {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "unavailable": 0,
            },
            "commodity_breakdown": {},
            "success": True,
        }

        raw_records = self.load_raw_records(src_path)
        summary["total_read"] = len(raw_records)

        if not raw_records:
            logger.info("No records found to process.")
            summary["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return summary

        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Enriching %d records with sentiment model '%s'...",
            len(raw_records),
            self.config.model_name,
        )

        # Batch process records
        enriched_records = self.analyzer.enrich_records(raw_records)

        # Write to destination processed file
        with dst_path.open("w", encoding="utf-8") as f:
            for rec in enriched_records:
                line = json.dumps(rec, ensure_ascii=False)
                f.write(line + "\n")

                summary["total_enriched"] += 1
                label = rec.get("sentiment_label", "unavailable")
                summary["sentiment_counts"][label] = summary["sentiment_counts"].get(label, 0) + 1

                comm = rec.get("commodity", "unknown")
                if comm not in summary["commodity_breakdown"]:
                    summary["commodity_breakdown"][comm] = {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                        "unavailable": 0,
                    }
                summary["commodity_breakdown"][comm][label] = (
                    summary["commodity_breakdown"][comm].get(label, 0) + 1
                )

        summary["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(
            "Sentiment processing completed: %d enriched records written to %s",
            summary["total_enriched"],
            dst_path,
        )

        return summary


def main() -> None:
    """Command-line entrypoint for sentiment enrichment."""
    parser = argparse.ArgumentParser(
        description="Enrich commodity news records with FinBERT sentiment analysis."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/commodity_news.jsonl",
        help="Source raw news JSONL file path (default: data/raw/commodity_news.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/commodity_news_sentiment.jsonl",
        help="Destination processed JSONL file path (default: data/processed/commodity_news_sentiment.jsonl)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ProsusAI/finbert",
        help="Hugging Face model identifier (default: ProsusAI/finbert)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on: 'cpu', 'cuda', or auto-detect (default: auto)",
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

    config = SentimentConfig(
        model_name=args.model_name,
        input_path=Path(args.input),
        output_path=Path(args.output),
        batch_size=args.batch_size,
    )
    if args.device:
        config.device = args.device

    pipeline = SentimentPipeline(config=config)
    result = pipeline.run(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )

    print("\n--- Sentiment Pipeline Execution Summary ---")
    print(f"Status:            {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Model:             {result['model_name']}")
    print(f"Input File:        {result['input_file']}")
    print(f"Output File:       {result['output_file']}")
    print(f"Total Read:        {result['total_read']}")
    print(f"Total Enriched:    {result['total_enriched']}")
    print("Sentiment Counts:")
    for lbl, cnt in result["sentiment_counts"].items():
        print(f"  - {lbl.capitalize()}: {cnt}")
    if result["commodity_breakdown"]:
        print("Commodity Breakdown:")
        for comm, stats in result["commodity_breakdown"].items():
            print(f"  [{comm}] Pos: {stats.get('positive', 0)}, Neg: {stats.get('negative', 0)}, Neu: {stats.get('neutral', 0)}, Unavail: {stats.get('unavailable', 0)}")


if __name__ == "__main__":
    main()
