"""
Formula 1 Lap Time Prediction - Inference & Prediction Module
Provides real-time next lap time forecasting using the trained LSTM model.
Supports historical race retrieval and interactive custom lap scenario prediction.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_processing import (
    DATA_DIR,
    MODELS_DIR,
    DEFAULT_FEATURE_COLUMNS,
    format_ms,
    time_to_ms,
    load_scalers,
    transform_data,
    inverse_transform_target,
    load_raw_datasets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_CACHE = None
_SCALERS_CACHE = None
_DATA_CACHE = None


def get_inference_engine(models_dir: str = MODELS_DIR) -> Tuple[tf.keras.Model, Any, Any]:
    """
    Load and cache the trained LSTM model and scalers for high-speed inference.
    """
    global _MODEL_CACHE, _SCALERS_CACHE
    model_path = os.path.join(models_dir, "lstm_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train the model first."
        )

    if _MODEL_CACHE is None:
        logger.info(f"Loading trained LSTM model from {model_path}...")
        _MODEL_CACHE = load_model(model_path, compile=False)

    if _SCALERS_CACHE is None:
        logger.info(f"Loading scalers from {models_dir}...")
        _SCALERS_CACHE = load_scalers(models_dir)

    return _MODEL_CACHE, _SCALERS_CACHE[0], _SCALERS_CACHE[1]


def get_cached_metadata(data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load raw reference datasets for drivers, races, and circuits.
    """
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = load_raw_datasets(data_dir)
    return _DATA_CACHE


def get_driver_list(min_year: int = 2018) -> List[Dict[str, Any]]:
    """
    Return list of drivers who competed in recent F1 seasons with details.
    """
    data = get_cached_metadata()
    races = data["races"]
    results = data["results"]
    drivers = data["drivers"]

    recent_race_ids = races[races["year"] >= min_year]["raceId"].unique()
    active_driver_ids = results[results["raceId"].isin(recent_race_ids)]["driverId"].unique()

    active_drivers = drivers[drivers["driverId"].isin(active_driver_ids)].copy()
    active_drivers["full_name"] = active_drivers["forename"] + " " + active_drivers["surname"]
    active_drivers = active_drivers.sort_values("surname")

    return active_drivers[["driverId", "driverRef", "code", "full_name", "nationality"]].to_dict("records")


def get_race_list(year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return list of races, optionally filtered by season year.
    """
    data = get_cached_metadata()
    races = data["races"]
    circuits = data["circuits"]

    merged = races.merge(circuits[["circuitId", "name"]].rename(columns={"name": "circuit_name"}), on="circuitId")
    if year:
        merged = merged[merged["year"] == year]

    merged = merged.sort_values(by=["year", "round"], ascending=[False, True])
    return merged[["raceId", "year", "round", "circuitId", "name", "circuit_name", "date"]].to_dict("records")


def predict_next_lap(
    recent_laps_ms: List[float],
    current_lap: int,
    position: int = 5,
    circuit_id: int = 1,
    driver_id: int = 1,
    constructor_id: int = 1,
    grid: int = 5,
    models_dir: str = MODELS_DIR,
) -> Dict[str, Any]:
    """
    Generate next lap time prediction using genuine LSTM model.
    Inputs:
      recent_laps_ms: List of 5 previous lap times in milliseconds.
      current_lap: The current lap number (target lap will be current_lap + 1).
    """
    if len(recent_laps_ms) != 5:
        raise ValueError(f"Sequence length must be 5 laps, got {len(recent_laps_ms)}")

    model, scaler_X, scaler_y = get_inference_engine(models_dir)

    # Construct the sequence matrix (5 timesteps, 7 features)
    # Features: ['milliseconds', 'lap', 'position', 'circuitId', 'driverId', 'constructorId', 'grid']
    seq_features = []
    for idx, lap_time in enumerate(recent_laps_ms):
        lap_num = current_lap - 4 + idx
        seq_features.append([
            float(lap_time),
            float(lap_num),
            float(position),
            float(circuit_id),
            float(driver_id),
            float(constructor_id),
            float(grid),
        ])

    X_input = np.array([seq_features], dtype=np.float32)  # shape: (1, 5, 7)

    # Scale input
    X_scaled, _ = transform_data(X_input, None, scaler_X, None)

    # Run LSTM inference
    y_scaled_pred = model.predict(X_scaled, verbose=0)

    # Inverse transform
    y_pred_ms = float(inverse_transform_target(y_scaled_pred, scaler_y)[0])
    target_lap = current_lap + 1

    return {
        "predicted_lap": target_lap,
        "predicted_ms": round(y_pred_ms),
        "predicted_formatted": format_ms(y_pred_ms),
        "predicted_seconds": round(y_pred_ms / 1000.0, 3),
        "input_sequence_ms": recent_laps_ms,
        "input_sequence_formatted": [format_ms(x) for x in recent_laps_ms],
        "delta_vs_previous_ms": round(y_pred_ms - recent_laps_ms[-1]),
        "delta_vs_previous_sec": round((y_pred_ms - recent_laps_ms[-1]) / 1000.0, 3),
    }


def predict_from_historical_race(
    race_id: int,
    driver_id: int,
    target_lap: int,
    sequence_length: int = 5,
    models_dir: str = MODELS_DIR,
) -> Dict[str, Any]:
    """
    Retrieve real race history for a given race and driver, extract the 5 preceding laps,
    run the trained LSTM prediction, and compare with the actual historical lap time.
    """
    data = get_cached_metadata()
    laps_df = data["lap_times"]
    races_df = data["races"]
    results_df = data["results"]
    drivers_df = data["drivers"]
    circuits_df = data["circuits"]

    driver_race_laps = laps_df[
        (laps_df["raceId"] == race_id) & (laps_df["driverId"] == driver_id)
    ].sort_values("lap").reset_index(drop=True)

    if driver_race_laps.empty:
        raise ValueError(f"No lap records found for Race ID {race_id} and Driver ID {driver_id}")

    # Check if target lap exists
    target_row = driver_race_laps[driver_race_laps["lap"] == target_lap]
    has_actual = not target_row.empty
    actual_ms = float(target_row["milliseconds"].iloc[0]) if has_actual else None

    # Get preceding sequence_length laps
    preceding_laps = driver_race_laps[driver_race_laps["lap"] < target_lap].tail(sequence_length)
    if len(preceding_laps) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} preceding laps before lap {target_lap}. "
            f"Only {len(preceding_laps)} laps available."
        )

    # Extract metadata
    race_info = races_df[races_df["raceId"] == race_id].iloc[0]
    circuit_id = int(race_info["circuitId"])
    circuit_name = circuits_df[circuits_df["circuitId"] == circuit_id]["name"].iloc[0]

    driver_info = drivers_df[drivers_df["driverId"] == driver_id].iloc[0]
    driver_name = f"{driver_info['forename']} {driver_info['surname']}"

    result_match = results_df[
        (results_df["raceId"] == race_id) & (results_df["driverId"] == driver_id)
    ]
    grid = int(result_match["grid"].iloc[0]) if not result_match.empty else 10
    constructor_id = int(result_match["constructorId"].iloc[0]) if not result_match.empty else 1

    last_known_lap = int(preceding_laps["lap"].iloc[-1])
    recent_laps_ms = [float(x) for x in preceding_laps["milliseconds"].values]
    last_position = int(preceding_laps["position"].iloc[-1])

    # Run model prediction
    pred_res = predict_next_lap(
        recent_laps_ms=recent_laps_ms,
        current_lap=last_known_lap,
        position=last_position,
        circuit_id=circuit_id,
        driver_id=driver_id,
        constructor_id=constructor_id,
        grid=grid,
        models_dir=models_dir,
    )

    pred_res.update({
        "race_name": race_info["name"],
        "race_year": int(race_info["year"]),
        "circuit_name": circuit_name,
        "driver_name": driver_name,
        "grid": grid,
        "has_actual": has_actual,
        "actual_ms": round(actual_ms) if actual_ms is not None else None,
        "actual_formatted": format_ms(actual_ms) if actual_ms is not None else "N/A",
        "actual_seconds": round(actual_ms / 1000.0, 3) if actual_ms is not None else None,
    })

    if has_actual:
        error_ms = round(actual_ms - pred_res["predicted_ms"])
        pred_res["error_ms"] = error_ms
        pred_res["error_seconds"] = round(error_ms / 1000.0, 3)
        pred_res["abs_error_ms"] = abs(error_ms)
        pred_res["accuracy_percent"] = round(100.0 - (abs(error_ms) / actual_ms * 100.0), 2)

    return pred_res
