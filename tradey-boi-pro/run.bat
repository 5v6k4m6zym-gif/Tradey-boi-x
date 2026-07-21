@echo off
cd /d "%~dp0"
title Tradey Boi Pro
color 0A

:: Activate venv if it exists, otherwise use system Python
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Starting Tradey Boi Pro...
start /b cmd /c "timeout /t 2 >nul && start http://localhost:8502"

streamlit run pro_dashboard.py ^
    --server.port 8502 ^
    --server.headless true ^
    --server.address 0.0.0.0 ^
    --browser.gatherUsageStats false ^
    --theme.base dark
