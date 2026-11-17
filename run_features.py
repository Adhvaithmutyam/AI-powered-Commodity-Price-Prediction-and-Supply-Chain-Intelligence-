#!/usr/bin/env python
"""
Entrypoint script for building standardized, timestamp-aware NLP feature datasets.
Usage:
    python run_features.py
    python run_features.py --cutoff 2026-10-20T12:00:00Z --windows 1,5
    python run_features.py --help
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
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

from src.features.pipeline import main

if __name__ == "__main__":
    main()
