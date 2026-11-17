#!/usr/bin/env python
"""
Entrypoint script for the NLP Evaluation Framework and Data-Quality Validation.

Usage:
    # 1. Run Data-Quality Validation on article features
    python run_evaluation.py --mode validate
    python run_evaluation.py --mode validate --input data/processed/nlp_article_features.jsonl

    # 2. Generate unbiased human annotation sample template
    python run_evaluation.py --mode sample --sample-size 50 --seed 42
    python run_evaluation.py --mode sample --input data/processed/nlp_article_features.jsonl --output data/evaluation/annotation_sample.jsonl

    # 3. Evaluate supervised classification against human-labeled ground truth
    python run_evaluation.py --mode evaluate --ground-truth data/evaluation/ground_truth.jsonl
    python run_evaluation.py --mode evaluate --input data/processed/nlp_article_features.jsonl --ground-truth data/evaluation/ground_truth.jsonl

    # Display command-line options
    python run_evaluation.py --help
"""

from pathlib import Path
import sys

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

from src.evaluation.cli import run_cli


if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
