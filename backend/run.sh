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

echo "Starting MCP Servers in the background..."
.venv/bin/python mcp_server.py &
MCP_PID1=$!
.venv/bin/python external_weather_mcp.py &
MCP_PID2=$!

# Trap SIGINT and SIGTERM to kill background processes when stopping
trap "kill $MCP_PID1 $MCP_PID2 2>/dev/null" EXIT INT TERM

# Wait a couple of seconds for the MCP servers to initialize
sleep 2

echo "Starting Demo 4 FastAPI Backend on http://0.0.0.0:8000..."
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
