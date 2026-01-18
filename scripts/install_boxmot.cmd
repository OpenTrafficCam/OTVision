@echo off
REM BOXMOT Installation Script for OTVision (Windows)
REM ==================================================
REM This script installs BOXMOT multi-object tracking dependencies
REM and optionally downloads ReID weights for appearance-based trackers.
REM
REM Usage:
REM   scripts\install_boxmot.cmd              Install BOXMOT dependencies only
REM   scripts\install_boxmot.cmd --with-reid  Also download ReID weights
REM
REM Requirements:
REM   - Python 3.12
REM   - uv package manager

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."
set "PROJECT_DIR=%cd%"

echo.
echo ========================================
echo  BOXMOT Installation for OTVision
echo ========================================
echo.
echo Working directory: %PROJECT_DIR%
echo.

REM Parse arguments
set "WITH_REID=false"
if "%1"=="--with-reid" set "WITH_REID=true"
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

REM Step 1: Check Python version
echo [INFO] Checking Python version...
FOR /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO SET "PYTHON_VERSION=%%F"
echo %PYTHON_VERSION%

if "x%PYTHON_VERSION:3.12=%"=="x%PYTHON_VERSION%" (
    echo [WARN] Python 3.12 is required but different version detected.
    echo [WARN] BOXMOT may not work correctly.
) else (
    echo [OK] Python 3.12 detected
)
echo.

REM Step 2: Check uv availability
echo [INFO] Checking uv package manager...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARN] uv not found. Installing uv...
    powershell -ExecutionPolicy Bypass -Command "& {irm https://astral.sh/uv/install.ps1 | iex}"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

    where uv >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install uv. Please install manually.
        exit /b 1
    )
)
echo [OK] uv package manager is available
echo.

REM Step 3: Install BOXMOT dependencies
echo ========================================
echo  Installing BOXMOT Dependencies
echo ========================================
echo.
echo [INFO] Running: uv pip install -e .[tracking_boxmot]
echo.

uv pip install -e ".[tracking_boxmot]"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install BOXMOT dependencies
    exit /b 1
)
echo [OK] BOXMOT dependencies installed successfully
echo.

REM Step 4: Verify installation
echo ========================================
echo  Verifying Installation
echo ========================================
echo.
echo [INFO] Testing BOXMOT import...

python -c "from boxmot import TRACKERS; print(f'Available trackers: {TRACKERS}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] BOXMOT import failed. Installation may be incomplete.
    exit /b 1
)
echo [OK] BOXMOT is properly installed
echo.

echo [INFO] Testing OTVision BOXMOT adapter...
python -c "from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter; print('BoxmotTrackerAdapter loaded successfully')" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] OTVision BOXMOT adapter could not be loaded. Check for errors.
) else (
    echo [OK] OTVision BOXMOT adapter is working
)
echo.

REM Step 5: Optionally download ReID weights
if "%WITH_REID%"=="true" (
    echo ========================================
    echo  Downloading ReID Weights
    echo ========================================
    echo.

    set "WEIGHTS_DIR=%PROJECT_DIR%\weights"
    if not exist "!WEIGHTS_DIR!" mkdir "!WEIGHTS_DIR!"

    set "REID_URL=https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.pt"
    set "REID_FILE=!WEIGHTS_DIR!\osnet_x0_25_msmt17.pt"

    if exist "!REID_FILE!" (
        echo [INFO] ReID weights already exist at: !REID_FILE!
    ) else (
        echo [INFO] Downloading OSNet ReID weights...
        powershell -Command "Invoke-WebRequest -Uri '!REID_URL!' -OutFile '!REID_FILE!'"
        if %errorlevel% neq 0 (
            echo [WARN] Failed to download ReID weights. You can download manually from:
            echo        !REID_URL!
        ) else (
            echo [OK] ReID weights downloaded to: !REID_FILE!
        )
    )
    echo.
    echo [INFO] To use appearance-based trackers ^(BotSORT, BoostTrack, etc.^),
    echo [INFO] add to your config:
    echo.
    echo TRACK:
    echo   BOXMOT:
    echo     ENABLED: true
    echo     TRACKER_TYPE: "botsort"
    echo     REID_WEIGHTS: "!REID_FILE!"
    echo.
)

REM Print summary
echo ========================================
echo  Installation Complete
echo ========================================
echo.
echo Available BOXMOT Trackers:
echo.
echo   Motion-Only ^(fast, no ReID weights needed^):
echo     - bytetrack  : High FPS, recommended for CPU
echo     - ocsort     : Alternative motion-only tracker
echo.
echo   Appearance-Based ^(higher accuracy, requires ReID weights^):
echo     - botsort    : Best overall accuracy
echo     - boosttrack : High identity consistency
echo     - strongsort : Balanced performance
echo     - deepocsort : Enhanced OcSORT with ReID
echo     - hybridsort : Hybrid motion-appearance
echo.
echo Quick Start:
echo   1. Copy boxmot_config.example.yaml to user_config.otvision.yaml
echo   2. Enable BOXMOT by setting TRACK.BOXMOT.ENABLED: true
echo   3. Run tracking: uv run track.py --paths /path/to/*.otdet
echo.
echo Documentation: BOXMOT_INTEGRATION.md
echo.

if "%WITH_REID%"=="false" (
    echo [INFO] Tip: Run with --with-reid to download ReID weights for appearance trackers
)
echo.
echo [OK] BOXMOT installation completed successfully!
echo.
goto :eof

:help
echo BOXMOT Installation Script for OTVision
echo.
echo Usage:
echo   %~nx0              Install BOXMOT dependencies only
echo   %~nx0 --with-reid  Also download ReID weights
echo   %~nx0 --help       Show this help message
goto :eof
