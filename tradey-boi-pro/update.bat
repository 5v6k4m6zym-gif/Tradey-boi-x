@echo off
REM Tradey Boi Pro — in-place updater (Windows)
REM Run this from inside your TradeyBoiPro folder instead of deleting and reinstalling.

set RELEASE_URL=https://github.com/5v6k4m6zym-gif/Tradey-boi-x/releases/latest/download/TradeyBoiPro.zip
set TMP_ZIP=%TEMP%\TradeyBoiPro_update.zip
set TMP_DIR=%TEMP%\TradeyBoiPro_update

echo.
echo  =========================================
echo    Tradey Boi Pro -- Updater
echo  =========================================
echo.

REM ── Stop running instance ──────────────────────────────────────────────────
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Tradey Boi Pro*" >nul 2>&1
timeout /t 1 >nul

REM ── Download latest release ────────────────────────────────────────────────
echo  Downloading latest release...
powershell -Command "& { $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%RELEASE_URL%' -OutFile '%TMP_ZIP%' }"
if errorlevel 1 (
    echo  ERROR: Download failed. Check your internet connection.
    pause
    exit /b 1
)
echo  Download complete.
echo.

REM ── Preserve user data before extraction ──────────────────────────────────
echo  Preserving your data...

REM Back up database
if exist "data\pro.db" (
    copy /Y "data\pro.db" "%TEMP%\pro.db.bak" >nul
)

REM Back up learned thresholds
if exist "config\adaptive_thresholds.json" (
    copy /Y "config\adaptive_thresholds.json" "%TEMP%\adaptive_thresholds.json.bak" >nul
)

REM Back up sweep winner
if exist "stop_sweep_winner.json" (
    copy /Y "stop_sweep_winner.json" "%TEMP%\stop_sweep_winner.json.bak" >nul
)
if exist "sweep_winner.json" (
    copy /Y "sweep_winner.json" "%TEMP%\sweep_winner.json.bak" >nul
)

REM ── Extract update ─────────────────────────────────────────────────────────
echo  Installing update...
powershell -Command "& { $ProgressPreference='SilentlyContinue'; Expand-Archive -Path '%TMP_ZIP%' -DestinationPath '.' -Force }"
del /Q "%TMP_ZIP%"

REM ── Restore user data ──────────────────────────────────────────────────────
if exist "%TEMP%\pro.db.bak" (
    copy /Y "%TEMP%\pro.db.bak" "data\pro.db" >nul
    del "%TEMP%\pro.db.bak"
)
if exist "%TEMP%\adaptive_thresholds.json.bak" (
    copy /Y "%TEMP%\adaptive_thresholds.json.bak" "config\adaptive_thresholds.json" >nul
    del "%TEMP%\adaptive_thresholds.json.bak"
)
if exist "%TEMP%\stop_sweep_winner.json.bak" (
    copy /Y "%TEMP%\stop_sweep_winner.json.bak" "stop_sweep_winner.json" >nul
    del "%TEMP%\stop_sweep_winner.json.bak"
)
if exist "%TEMP%\sweep_winner.json.bak" (
    copy /Y "%TEMP%\sweep_winner.json.bak" "sweep_winner.json" >nul
    del "%TEMP%\sweep_winner.json.bak"
)

echo  Files updated.
echo.

REM ── Update dependencies ────────────────────────────────────────────────────
echo  Updating dependencies...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt -q --disable-pip-version-check
    echo  Dependencies up to date.
) else (
    echo  No .venv found -- run install.bat first if this is a fresh install.
)
echo.

echo  Update complete! Run install.bat or run.bat to start the dashboard.
echo.
pause
