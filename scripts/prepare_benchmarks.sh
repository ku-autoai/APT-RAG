#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly MAIN_SCRIPT="$APT_ROOT/src/artifacts_construction/benchmarks/prepare_benchmarks.py"

exec "$PYTHON_BIN" "$MAIN_SCRIPT" --repo-root "$APT_ROOT" "$@"
