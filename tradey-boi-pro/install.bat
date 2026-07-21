@echo off
cd /d "%~dp0"
title Tradey Boi Pro — Installer
color 0A

echo.
echo  =========================================
echo    Tradey Boi Pro — Windows Installer
echo  =========================================
echo.

:: ── Check Python (try python, then py launcher) ───────────────────────────────
set PYCMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo  ERROR: Python not found.
        echo.
        echo  Please install Python 3.10 or newer from:
        echo  https://www.python.org/downloads/
        echo.
        echo  IMPORTANT: Tick "Add Python to PATH" during install.
        echo.
        pause
        start https://www.python.org/downloads/
        exit /b 1
    )
    set PYCMD=py
)

for /f "tokens=2 delims= " %%v in ('%PYCMD% --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%
echo.

:: ── Create virtual environment ────────────────────────────────────────────────
if not exist ".venv" (
    echo  Creating virtual environment...
    %PYCMD% -m venv .venv
    if %errorlevel% neq 0 (
        echo  ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
    echo  Done.
    echo.
)

:: ── Activate venv ─────────────────────────────────────────────────────────────
call .venv\Scripts\activate.bat

:: ── Install dependencies ──────────────────────────────────────────────────────
echo  Installing dependencies (this may take a minute)...
pip install -r requirements.txt -q --disable-pip-version-check
if %errorlevel% neq 0 (
    echo  ERROR: Failed to install dependencies.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)
echo  Dependencies installed.
echo.

:: ── Launch ───────────────────────────────────────────────────────────────────
echo  Starting Tradey Boi Pro...
echo  Dashboard will open in your browser at http://localhost:8502
echo.
echo  (Close this window to stop the bot)
echo.

:: Open browser after 3 seconds
start /b cmd /c "timeout /t 3 >nul && start http://localhost:8502"

:: Start the dashboard
streamlit run pro_dashboard.py ^
    --server.port 8502 ^
    --server.headless true ^
    --server.address 0.0.0.0 ^
    --browser.gatherUsageStats false ^
    --theme.base dark

pause
