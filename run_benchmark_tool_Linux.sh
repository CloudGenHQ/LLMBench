#!/usr/bin/env bash
# ──────────────────────────────────────────────
#  LLM Benchmark Tool – Linux launcher
# ──────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: '$PYTHON' not found. Install Python 3.8+ first." >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/llm_benchmark.py" "$@"
