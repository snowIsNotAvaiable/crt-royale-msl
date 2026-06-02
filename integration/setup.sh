#!/usr/bin/env bash
# Thin wrapper around setup.py for Mac/Linux convenience.
# All real logic lives in setup.py (cross-platform).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/setup.py" "$@"
