"""
Formula 1 Lap Time Prediction - Evaluation Module
Calculates MAE, MSE, RMSE, R2-score, residuals, and actual vs predicted comparisons.
Exports metrics to outputs/evaluation_results.json and outputs/test_predictions.csv.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple, Optional

# Set local cache paths before importing TF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processing import (
    MODELS_DIR,
    OUTPUTS_DIR,
    format_ms,
    load_scalers,
    transform_data,
    inverse_transform_target,
    ensure_directories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_pipeline(
    model_path: Optional[str] = None,
    models_dir: str = MODELS_DIR,
    outputs_dir: str = OUTPUTS_DIR,
    sample_size: int = 5000,
) -> Dict[str, Any]:
    """
    Load saved model, test sequences, and scalers; compute evaluation metrics;
    and export evaluation results.
    """
    ensure_directories()
    if model_path is None:
        model_path = os.path.join(models_dir, "lstm_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model.py first.")

    # Load test arrays
    x_test_path = os.path.join(outputs_dir, "X_test.npy")
    y_test_path = os.path.join(outputs_dir, "y_test.npy")
    meta_test_path = os.path.join(outputs_dir, "meta_test.csv")

    if not (os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        raise FileNotFoundError(
            f"Test sequence arrays not found in {outputs_dir}. Run train_model.py first."
        )

    logger.info("Loading test dataset arrays and scalers...")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    meta_df = pd.read_csv(meta_test_path) if os.path.exists(meta_test_path) else pd.DataFrame()

    scaler_X, scaler_y = load_scalers(models_dir)

    logger.info(f"Loading trained LSTM model from {model_path}...")
    model = load_model(model_path, compile=False)

    # Scale test input features
    X_test_scaled, _ = transform_data(X_test, None, scaler_X, None)

    logger.info(f"Generating predictions for {len(X_test)} test sequence samples...")
    y_pred_scaled = model.predict(X_test_scaled, batch_size=256, verbose=1).flatten()

    # Inverse transform predictions to raw milliseconds
    y_pred_ms = inverse_transform_target(y_pred_scaled, scaler_y)
    y_true_ms = y_test

    # Compute key metrics in milliseconds
    mae_ms = float(mean_absolute_error(y_true_ms, y_pred_ms))
    mse_ms = float(mean_squared_error(y_true_ms, y_pred_ms))
    rmse_ms = float(np.sqrt(mse_ms))
    r2 = float(r2_score(y_true_ms, y_pred_ms))

    # Convert to seconds for human readability
    mae_sec = mae_ms / 1000.0
    rmse_sec = rmse_ms / 1000.0
    mean_lap_sec = float(np.mean(y_true_ms)) / 1000.0
    percentage_error = float(mae_ms / np.mean(y_true_ms) * 100.0)

    # Calculate residuals
    residuals = y_true_ms - y_pred_ms
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))

    results = {
        "mae_ms": round(mae_ms, 2),
        "mse_ms": round(mse_ms, 2),
        "rmse_ms": round(rmse_ms, 2),
        "mae_sec": round(mae_sec, 3),
        "rmse_sec": round(rmse_sec, 3),
        "r2_score": round(r2, 4),
        "mean_actual_sec": round(mean_lap_sec, 3),
        "mean_absolute_percentage_error": round(percentage_error, 2),
        "residuals_mean_ms": round(res_mean, 2),
        "residuals_std_ms": round(res_std, 2),
        "total_test_samples": int(len(y_test)),
        "formatted_mae": f"{mae_sec:.3f} s ({mae_ms:.1f} ms)",
        "formatted_rmse": f"{rmse_sec:.3f} s ({rmse_ms:.1f} ms)",
    }

    # Save metrics JSON
    results_json_path = os.path.join(outputs_dir, "evaluation_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation metrics saved to {results_json_path}")

    # Build predictions comparison DataFrame
    pred_df = meta_df.copy() if not meta_df.empty else pd.DataFrame({"actual_ms": y_true_ms})
    pred_df["actual_ms"] = y_true_ms
    pred_df["predicted_ms"] = np.round(y_pred_ms).astype(int)
    pred_df["error_ms"] = np.round(residuals).astype(int)
    pred_df["abs_error_ms"] = np.abs(pred_df["error_ms"])
    pred_df["error_sec"] = np.round(residuals / 1000.0, 3)
    pred_df["abs_error_sec"] = np.round(pred_df["abs_error_ms"] / 1000.0, 3)
    pred_df["actual_formatted"] = pred_df["actual_ms"].apply(format_ms)
    pred_df["predicted_formatted"] = pred_df["predicted_ms"].apply(format_ms)

    # Save test predictions CSV
    pred_csv_path = os.path.join(outputs_dir, "test_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    logger.info(f"Predictions saved to {pred_csv_path}")

    return {
        "metrics": results,
        "predictions_df": pred_df,
        "results_path": results_json_path,
        "predictions_path": pred_csv_path,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("FORMULA 1 LAP TIME PREDICTION - MODEL EVALUATION")
    print("=" * 60)
    eval_res = evaluate_model_pipeline()
    m = eval_res["metrics"]
    print("Model Evaluation Summary:")
    print(f"  • Mean Absolute Error (MAE): {m['mae_ms']:.2f} ms ({m['mae_sec']:.3f} seconds)")
    print(f"  • Root Mean Squared Error (RMSE): {m['rmse_ms']:.2f} ms ({m['rmse_sec']:.3f} seconds)")
    print(f"  • Mean Squared Error (MSE): {m['mse_ms']:.2f}")
    print(f"  • R² Score: {m['r2_score']:.4f}")
    print(f"  • Mean Relative Error: {m['mean_absolute_percentage_error']:.2f}%")
    print(f"  • Total Test Samples: {m['total_test_samples']}")
    print("=" * 60)
