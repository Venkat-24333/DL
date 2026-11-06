@echo off
title F1 Lap Time Prediction System - Deep Learning LSTM
color 0C

echo =====================================================================
echo       FORMULA 1 LAP TIME PREDICTION SYSTEM (DEEP LEARNING LSTM)
echo =====================================================================
echo.

:: Navigate to script directory
cd /d "%~dp0"

:: Set local cache environments to prevent permission issues
set "KERAS_HOME=%~dp0.keras"
set "MPLCONFIGDIR=%~dp0.mpl_cache"
set "TF_ENABLE_ONEDNN_OPTS=0"

:: Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in system PATH.
    echo Please install Python 3.10-3.12 and add it to your PATH.
    echo.
    pause
    exit /b 1
)

:: Check if required files exist
if not exist "models\lstm_model.keras" (
    echo [NOTICE] Trained LSTM model not found in models\ directory.
    echo Training model now... This will take about 1 minute.
    echo.
    python train_model.py --epochs 15 --batch-size 128 --seq-len 5
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    python evaluation.py
)

echo [OK] Environment verified.
echo [OK] Model ready.
echo.
echo Launching Streamlit web dashboard at http://localhost:8501 ...
echo (Press Ctrl+C in this window anytime to stop the server)
echo =====================================================================
echo.

:: Launch Streamlit app
python -m streamlit run app.py --server.port 8501

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Application terminated with code %errorlevel%.
    pause
)
