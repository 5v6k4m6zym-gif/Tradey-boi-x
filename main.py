"""
Tradey Boi X — Production entrypoint.
Runs the background scanner and Streamlit dashboard together.
"""
import subprocess
import sys
import os

# Start scanner in background
scanner = subprocess.Popen([sys.executable, "tradey-boi-x/scanner.py"])
print(f"Scanner started (PID {scanner.pid})")

# Start Streamlit dashboard in foreground
port = os.environ.get("PORT", "5000")
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "tradey-boi-x/dashboard.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
])

# If dashboard exits, stop scanner
scanner.terminate()
