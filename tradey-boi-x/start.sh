#!/bin/bash
# Production startup — runs scanner in background, dashboard in foreground
echo "Starting Tradey Boi X..."

cd "$(dirname "$0")"

# Start the background scanner
python scanner.py &
SCANNER_PID=$!
echo "Scanner started (PID $SCANNER_PID)"

# Start the Streamlit dashboard (foreground — keeps container alive)
streamlit run dashboard.py --server.port "${PORT:-5000}" --server.address 0.0.0.0

# If dashboard exits, stop the scanner too
kill $SCANNER_PID 2>/dev/null
