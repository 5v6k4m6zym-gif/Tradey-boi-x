#!/usr/bin/env bash
set -e

echo ""
echo " ========================================="
echo "   Tradey Boi Pro — Mac/Linux Installer"
echo " ========================================="
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo " ERROR: Python 3 not found."
    echo ""
    echo " Install it with:"
    echo "   Mac:   brew install python3"
    echo "   Linux: sudo apt install python3 python3-venv python3-pip"
    echo ""
    exit 1
fi

PYVER=$(python3 --version 2>&1 | awk '{print $2}')
echo " Found Python $PYVER"
echo ""

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv .venv
    echo " Done."
    echo ""
fi

source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
echo " Installing dependencies..."
pip install -r requirements.txt -q --disable-pip-version-check
echo " Done."
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo " Starting Tradey Boi Pro..."
echo " Dashboard: http://localhost:8502"
echo " (Press Ctrl+C to stop)"
echo ""

# Open browser after 3 seconds (best-effort)
(sleep 3 && python3 -m webbrowser "http://localhost:8502") &

streamlit run pro_dashboard.py \
    --server.port 8502 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --theme.base dark
