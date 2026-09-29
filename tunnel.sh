#!/usr/bin/env bash
# Quick launcher to expose Demo 4 via Ngrok to a public HTTPS URL
set -e

if ! command -v ngrok >/dev/null 2>&1; then
  echo "❌ ngrok is not installed on this system."
  echo "Install it via: sudo snap install ngrok  OR  download from https://ngrok.com/download"
  exit 1
fi

echo "=========================================================="
echo "⚡ Starting Ngrok Tunnel for Demo 6 (Port 5177)..."
echo "👉 Pointing public tunnel to http://localhost:5177"
echo ""

ngrok http 5177
