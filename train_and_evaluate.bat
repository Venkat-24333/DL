@echo off
title F1 Model Training Pipeline - Deep Learning LSTM
color 0A

echo =====================================================================
echo    FORMULA 1 LAP TIME PREDICTION - RETRAIN LSTM DEEP LEARNING MODEL
echo =====================================================================
echo.

cd /d "%~dp0"

set "KERAS_HOME=%~dp0.keras"
set "MPLCONFIGDIR=%~dp0.mpl_cache"
set "TF_ENABLE_ONEDNN_OPTS=0"

echo Step 1: Preprocessing telemetry and training LSTM model...
python train_model.py --epochs 15 --batch-size 128 --seq-len 5
if %errorlevel% neq 0 (
    echo [ERROR] Training failed.
    pause
    exit /b 1
)

echo.
echo Step 2: Evaluating model on 2024 season test set...
python evaluation.py
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation failed.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] Training and evaluation completed successfully!
echo Model saved to: models\lstm_model.keras
echo Metrics saved to: outputs\evaluation_results.json
echo =====================================================================
echo.
pause
