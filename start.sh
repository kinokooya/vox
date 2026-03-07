#!/usr/bin/env bash
# Launch Vox from the project directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec .venv/Scripts/python.exe -m vox "$@"
