#!/usr/bin/env python
"""
Entrypoint script for running Supply-Chain Event Extraction on commodity news.
Usage:
    python run_event_extraction.py
    python run_event_extraction.py --help
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.event_extraction.pipeline import main

if __name__ == "__main__":
    main()
