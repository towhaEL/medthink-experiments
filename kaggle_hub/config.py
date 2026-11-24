"""
Configuration and system paths for Kaggle Hub.
Zero external dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base storage directory (overridable via KAGGLE_HUB_DIR environment variable)
KAGGLE_HUB_DIR = Path(
    os.environ.get("KAGGLE_HUB_DIR", Path.home() / ".kaggle_hub")
).resolve()
BASE_HUB_DIR = KAGGLE_HUB_DIR

ENV_DIR = KAGGLE_HUB_DIR / "envs"
STAGING_DIR = KAGGLE_HUB_DIR / "staging"
ACTIVE_SYMLINK = KAGGLE_HUB_DIR / "active"
ACCOUNTS_FILE = KAGGLE_HUB_DIR / "accounts.json"
JOBS_FILE = KAGGLE_HUB_DIR / "jobs.json"

# Package paths
PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_NOTEBOOKS_DIR = WORKSPACE_ROOT / "notebooks"
DEFAULT_OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

# Supported Kaggle Accelerators
ACCELERATORS: dict[str, str] = {
    "t4": "NvidiaTeslaT4",
    "t4x2": "NvidiaTeslaT4",
    "nvidiatetlat4": "NvidiaTeslaT4",
    "nvidiat4": "NvidiaTeslaT4",
    "p100": "NvidiaTeslaP100",
    "nvidiatestlap100": "NvidiaTeslaP100",
    "tpu": "TpuV5E8",
    "tpuv5e8": "TpuV5E8",
    "cpu": "None",
    "none": "None",
}

ACCELERATOR_CHOICES = ["None", "NvidiaTeslaT4", "NvidiaTeslaP100", "TpuV5E8"]

DEFAULT_PUSH_TIMEOUT = 3600
DEFAULT_POLL_INTERVAL = 10


def normalize_accelerator(acc: str | None) -> str:
    """Normalize user input or accelerator string to Kaggle CLI flag."""
    if not acc:
        return "NvidiaTeslaT4"
    cleaned = acc.strip().replace(" ", "").replace("-", "").lower()
    return ACCELERATORS.get(cleaned, acc)


def ensure_dirs(hub_dir: Path | None = None) -> None:
    """Ensure all required directories exist with safe permissions."""
    base = hub_dir or KAGGLE_HUB_DIR
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    (base / "envs").mkdir(mode=0o700, parents=True, exist_ok=True)
    (base / "staging").mkdir(mode=0o755, parents=True, exist_ok=True)
    DEFAULT_OUTPUTS_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
