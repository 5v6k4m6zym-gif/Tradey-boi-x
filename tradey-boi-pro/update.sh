#!/usr/bin/env bash
# Tradey Boi Pro — in-place updater (Mac / Linux)
# Run this from inside your TradeyBoiPro folder instead of deleting and reinstalling.
set -e

RELEASE_URL="https://github.com/5v6k4m6zym-gif/Tradey-boi-x/releases/latest/download/TradeyBoiPro.zip"
TMP_ZIP="/tmp/TradeyBoiPro_update.zip"

echo ""
echo " ========================================="
echo "   Tradey Boi Pro — Updater"
echo " ========================================="
echo ""

# ── Stop running instance ─────────────────────────────────────────────────────
if pgrep -f "streamlit run pro_dashboard" > /dev/null 2>&1; then
    echo " Stopping running dashboard..."
    pkill -f "streamlit run pro_dashboard" || true
    sleep 2
fi

# ── Download latest release ───────────────────────────────────────────────────
echo " Downloading latest release..."
if command -v curl &>/dev/null; then
    curl -L "$RELEASE_URL" -o "$TMP_ZIP" --progress-bar
elif command -v wget &>/dev/null; then
    wget -q --show-progress "$RELEASE_URL" -O "$TMP_ZIP"
else
    echo " ERROR: curl or wget is required."
    exit 1
fi
echo " Download complete."
echo ""

# ── Extract, preserving data and learned config ───────────────────────────────
echo " Installing update..."
# -o = overwrite without prompting
# -x = exclude these paths so user data is never overwritten
unzip -o "$TMP_ZIP" \
    -x "data/*" \
    -x "config/adaptive_thresholds.json" \
    -x ".venv/*" \
    -x "stop_sweep_winner.json" \
    -x "sweep_winner.json" \
    > /dev/null

rm -f "$TMP_ZIP"
echo " Files updated."
echo ""

# ── Update dependencies ───────────────────────────────────────────────────────
echo " Updating dependencies..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    pip install -r requirements.txt -q --disable-pip-version-check
    echo " Dependencies up to date."
else
    echo " No .venv found — run install.sh first if this is a fresh install."
fi
echo ""

echo " ✅ Update complete! Run ./run.sh (or install.sh) to start the dashboard."
echo ""
