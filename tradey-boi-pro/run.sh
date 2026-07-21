#!/usr/bin/env bash
# Quick-start after first install
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

(sleep 2 && python3 -m webbrowser "http://localhost:8502") &

streamlit run pro_dashboard.py \
    --server.port 8502 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --theme.base dark
