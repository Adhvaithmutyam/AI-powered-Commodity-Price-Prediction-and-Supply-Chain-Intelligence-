"""
Event extraction pipeline orchestrating record ingestion, classification, and JSONL persistence.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .base import BaseEventExtractor, EventRecord
from .config import EventConfig
from .rule_based import RuleBasedEventExtractor

logger = logging.getLogger(__name__)


class EventExtractionPipeline:
    """
    Orchestrates the extraction of supply-chain events from processed commodity news.
    """

    def __init__(
        self,
        config: Optional[EventConfig] = None,
        extractor: Optional[BaseEventExtractor] = None,
    ):
        self.config = config or EventConfig()
        self.extractor = extractor or RuleBasedEventExtractor()

    def resolve_input_path(self, input_path: Optional[Path]) -> Optional[Path]:
        """Locate the best available input file (primary or fallback)."""
        if input_path and Path(input_path).exists():
            return Path(input_path)

        if self.config.input_path.exists():
            return self.config.input_path

        if self.config.fallback_input_path.exists():
            logger.info(
                "Primary input %s not found. Falling back to %s",
                self.config.input_path,
                self.config.fallback_input_path,
            )
            return self.config.fallback_input_path

        return None

    def load_records(self, filepath: Path) -> List[Dict[str, Any]]:
        """Read and parse article records from a JSONL file."""
        records: List[Dict[str, Any]] = []
        if not filepath.exists():
            logger.warning("Input file does not exist: %s", filepath)
            return records

        with filepath.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    if isinstance(data, dict):
                        records.append(data)
                except json.JSONDecodeError as err:
                    logger.warning("Skipping invalid JSON on line %d of %s: %s", line_no, filepath, err)

        logger.info("Loaded %d records from %s", len(records), filepath)
        return records

    def run(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute event extraction and save structured results to JSONL.

        Returns:
            Dictionary summary with breakdown of extracted events and market impacts.
        """
        resolved_src = self.resolve_input_path(input_path)
        dest_path = Path(output_path or self.config.output_path)

        summary: Dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_file": str(resolved_src) if resolved_src else None,
            "output_file": str(dest_path),
            "total_read": 0,
            "total_extracted": 0,
            "event_type_counts": {},
            "impact_direction_counts": {},
            "commodity_counts": {},
            "status_counts": {},
            "success": True,
        }

        if not resolved_src:
            logger.error("No valid input file found to extract events from.")
            summary["success"] = False
            summary["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return summary

        records = self.load_records(resolved_src)
        summary["total_read"] = len(records)

        if not records:
            logger.info("No records to extract events from.")
            summary["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return summary

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting supply-chain events from %d records...", len(records))

        # Extract events
        event_records: List[EventRecord] = self.extractor.extract_events(records)

        # Write to JSONL
        with dest_path.open("w", encoding="utf-8") as f:
            for event in event_records:
                event_dict = event.to_dict()
                f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")

                summary["total_extracted"] += 1

                # Update counts
                etype = event.event_type
                summary["event_type_counts"][etype] = summary["event_type_counts"].get(etype, 0) + 1

                direction = event.impact_direction
                summary["impact_direction_counts"][direction] = summary["impact_direction_counts"].get(direction, 0) + 1

                comm = event.commodity
                summary["commodity_counts"][comm] = summary["commodity_counts"].get(comm, 0) + 1

                status = event.extraction_status
                summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

        summary["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(
            "Event extraction completed: %d events written to %s",
            summary["total_extracted"],
            dest_path,
        )

        return summary


def main() -> None:
    """Command-line entrypoint for event extraction."""
    parser = argparse.ArgumentParser(
        description="Extract supply-chain events and market implications from commodity news."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL path (defaults to data/processed/commodity_news_sentiment.jsonl or data/raw/commodity_news.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/commodity_events.jsonl",
        help="Destination JSONL path (default: data/processed/commodity_events.jsonl)",
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

    config = EventConfig(
        output_path=Path(args.output),
    )
    if args.input:
        config.input_path = Path(args.input)

    pipeline = EventExtractionPipeline(config=config)
    result = pipeline.run(
        input_path=Path(args.input) if args.input else None,
        output_path=Path(args.output),
    )

    print("\n--- Supply-Chain Event Extraction Summary ---")
    print(f"Status:             {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Input Source:       {result['input_file']}")
    print(f"Destination Output: {result['output_file']}")
    print(f"Total Read:         {result['total_read']}")
    print(f"Total Extracted:    {result['total_extracted']}")
    print("\nEvent Types:")
    for etype, cnt in result["event_type_counts"].items():
        print(f"  - {etype}: {cnt}")
    print("\nPotential Market Impact Direction:")
    for d, cnt in result["impact_direction_counts"].items():
        print(f"  - {d}: {cnt}")


if __name__ == "__main__":
    main()
