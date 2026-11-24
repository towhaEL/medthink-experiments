#!/usr/bin/env bash
# Kaggle Hub Launcher Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="$(which python3 || echo "python")"

if [ "$1" = "web" ] || [ "$1" = "serve" ] || [ "$1" = "ui" ] || [ -z "$1" ]; then
    shift 2>/dev/null || true
    echo "Starting Kaggle Hub Web Dashboard..."
    cd "$PARENT_DIR"
    exec "$PYTHON_BIN" -m kaggle_hub serve "$@"
else
    cd "$PARENT_DIR"
    exec "$PYTHON_BIN" -m kaggle_hub "$@"
fi
