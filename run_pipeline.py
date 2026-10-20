#!/usr/bin/env python
"""
Entrypoint script for running the News Ingestion Pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py --help
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.news_intelligence.pipeline import main

if __name__ == "__main__":
    main()
