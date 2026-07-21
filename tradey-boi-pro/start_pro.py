"""
Tradey Boi Pro — Launcher
Run this file to start the control centre:
    python start_pro.py
"""
import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent


def check_dependencies():
    missing = []
    for pkg in ["streamlit", "plotly", "ib_insync", "nest_asyncio",
                "yfinance", "pandas", "numpy", "pytz"]:
        try:
            __import__(pkg.replace("-", "_").replace("ib_insync", "ib_insync"))
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    print("=" * 60)
    print("  🤖  Tradey Boi Pro — Autonomous Trading Platform")
    print("=" * 60)
    print()

    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Installing now…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r",
             str(HERE / "requirements.txt"), "-q"]
        )
        print("✅ Dependencies installed")

    port = int(os.environ.get("PRO_PORT", "8502"))
    print(f"🚀 Starting dashboard on http://localhost:{port}")
    print("   Press Ctrl+C to stop")
    print()

    os.chdir(HERE)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "pro_dashboard.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
    ])


if __name__ == "__main__":
    main()
