#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
  echo "Initializing virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

echo "Starting Demo 4 FastAPI Backend on http://0.0.0.0:8000..."
exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
