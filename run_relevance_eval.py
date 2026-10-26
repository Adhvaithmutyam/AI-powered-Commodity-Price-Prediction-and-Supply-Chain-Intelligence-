#!/usr/bin/env python
"""
Entrypoint script for evaluating commodity news relevance.
Usage:
    python run_relevance_eval.py
    python run_relevance_eval.py --input data/raw/commodity_news.jsonl
    python run_relevance_eval.py --help
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.news_intelligence.relevance import (
    RelevanceClass,
    assess_relevance,
    enrich_record_with_relevance,
)

logger = logging.getLogger("relevance_eval")


def evaluate_file(
    input_path: Path,
    output_path: Path = None,
) -> Dict[str, Any]:
    """
    Evaluate news records from a JSONL file and optionally save enriched records.
    """
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return {}

    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                records.append(rec)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON on line %d", line_num)

    total = len(records)
    print(f"\n=======================================================")
    print(f" News Relevance Evaluation: {input_path}")
    print(f" Total records loaded: {total}")
    print(f"=======================================================")

    if total == 0:
        print("No records found to evaluate.")
        return {}

    counts: Dict[str, int] = {
        RelevanceClass.DIRECTLY_RELEVANT.value: 0,
        RelevanceClass.POTENTIALLY_RELEVANT.value: 0,
        RelevanceClass.IRRELEVANT.value: 0,
        RelevanceClass.UNCERTAIN.value: 0,
    }
    commodity_breakdown: Dict[str, Dict[str, int]] = {}
    enriched_records: List[Dict[str, Any]] = []
    category_samples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in counts}

    for rec in records:
        enriched = enrich_record_with_relevance(rec)
        enriched_records.append(enriched)

        r_class = enriched.get("relevance_class", RelevanceClass.UNCERTAIN.value)
        counts[r_class] = counts.get(r_class, 0) + 1

        comm = enriched.get("commodity", "unknown")
        if comm not in commodity_breakdown:
            commodity_breakdown[comm] = {k: 0 for k in counts}
        commodity_breakdown[comm][r_class] = commodity_breakdown[comm].get(r_class, 0) + 1

        if len(category_samples[r_class]) < 3:
            category_samples[r_class].append(enriched)

    print("\n--- Relevance Class Distribution ---")
    for r_class, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {r_class:<22}: {count:>4} ({pct:>5.1f}%)")

    print("\n--- Commodity Breakdown ---")
    for comm, class_counts in commodity_breakdown.items():
        print(f"  [{comm.upper()}]")
        for r_class, count in class_counts.items():
            if count > 0:
                print(f"    - {r_class}: {count}")

    print("\n--- Sample Evaluated Headlines ---")
    for r_class, samples in category_samples.items():
        if samples:
            print(f"\n[{r_class.upper()} SAMPLES]")
            for s in samples:
                title = s.get("title", "No Title")
                score = s.get("relevance_score", 0.0)
                indicators = s.get("matched_indicators", [])
                print(f"  * \"{title}\"")
                print(f"    Score: {score:.2f} | Indicators: {indicators}")
                print(f"    Rationale: {s.get('relevance_explanation')}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for enr in enriched_records:
                f.write(json.dumps(enr, ensure_ascii=False) + "\n")
        print(f"\nEnriched records saved to: {output_path}")

    return {
        "total": total,
        "counts": counts,
        "commodity_breakdown": commodity_breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate commodity news relevance against headline signals."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/commodity_news.jsonl",
        help="Input JSONL file (default: data/raw/commodity_news.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination JSONL file to save enriched relevance records",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    evaluate_file(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
