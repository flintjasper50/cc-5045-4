#!/usr/bin/env bash
# Convenience launcher for the multi-device render verification UI.
# No third-party packages required: Python 3 standard library only.
set -euo pipefail
cd "$(dirname "$0")"
HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"
echo "启动多端出图验证界面: http://${HOST}:${PORT}"
exec python3 backend/server.py "${HOST}" "${PORT}"
