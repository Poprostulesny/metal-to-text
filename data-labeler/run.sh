#!/usr/bin/env bash
# Painless launch: create an isolated venv on first run, then start the app.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "[labeler] Creating isolated venv..."
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements-labeler.txt
fi

echo "[labeler] Starting on http://127.0.0.1:8765"
( sleep 1; python3 -m webbrowser "http://127.0.0.1:8765" >/dev/null 2>&1 || true ) &
exec "$PY" -m server
