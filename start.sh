#!/usr/bin/env bash
# Standalone startup script for Demo 4 (Hybrid Knowledge AI)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "⚡ Starting Demo 4: Hybrid General + Organisational Knowledge"
echo "=========================================================="

# 1. Start Python FastAPI Backend (Port 8000)
echo "Starting Backend (FastAPI on http://127.0.0.1:8000)..."
cd "$DIR/backend"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
fi
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# 2. Start Frontend Dev Server (Port 5177)
echo "Starting Frontend (Vite React on http://127.0.0.1:5177)..."
cd "$DIR/frontend"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🚀 Demo 4 is running!"
echo "👉 Open Frontend: http://localhost:5177"
echo "👉 API Swagger Docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit 0" SIGINT SIGTERM EXIT
wait
