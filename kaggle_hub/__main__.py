"""
Entry point for running Kaggle Hub directly via:
    python3 -m kaggle_hub ...
"""

from __future__ import annotations

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
