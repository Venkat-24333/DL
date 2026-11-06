"""
Formula 1 Lap Time Prediction - Data Processing Module
Handles Kaggle dataset verification and download, merging, cleaning,
feature engineering, sequence generation, scaling, and train/test splitting.
"""

import os
import shutil
import json
import logging
from typing import Dict, Tuple, List, Optional, Any

# Configure local caches to avoid sandbox permission errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants and default paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

REQUIRED_CSV_FILES = [
    "lap_times.csv",
    "races.csv",
    "results.csv",
    "drivers.csv",
    "circuits.csv",
]

DEFAULT_FEATURE_COLUMNS = [
    "milliseconds",
    "lap",
    "position",
    "circuitId",
    "driverId",
    "constructorId",
    "grid",
]


def ensure_directories():
    """Ensure required project directories exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, os.path.join(BASE_DIR, ".keras"), os.path.join(BASE_DIR, ".mpl_cache")]:
        os.makedirs(d, exist_ok=True)


def format_ms(ms: Optional[float]) -> str:
    """
    Convert milliseconds into standard F1 lap time string format (m:ss.sss).
    Example: 84567 -> '1:24.567'
    """
    if ms is None or np.isnan(ms) or ms <= 0:
        return "N/A"
    ms_int = int(round(ms))
    minutes = ms_int // 60000
    seconds = (ms_int % 60000) // 1000
    millis = ms_int % 1000
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def time_to_ms(time_str: Optional[str]) -> Optional[int]:
    """
    Convert standard lap time string (e.g. '1:24.567') to milliseconds.
    """
    if not isinstance(time_str, str) or ":" not in time_str:
        return None
    try:
        parts = time_str.strip().split(":")
        minutes = int(parts[0])
        sec_parts = parts[1].split(".")
        seconds = int(sec_parts[0])
        millis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return minutes * 60000 + seconds * 1000 + millis
    except Exception:
        return None


def verify_dataset_files(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """
    Verify presence and basic statistics of required CSV files.
    """
    ensure_directories()
    results = {"valid": True, "files": {}, "missing_files": []}
    for filename in REQUIRED_CSV_FILES:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            file_size_bytes = os.path.getsize(filepath)
            results["files"][filename] = {
                "exists": True,
                "path": filepath,
                "size_mb": round(file_size_bytes / (1024 * 1024), 2),
            }
        else:
            results["valid"] = False
            results["missing_files"].append(filename)
            results["files"][filename] = {"exists": False, "path": filepath, "size_mb": 0}
    return results


def download_kaggle_dataset(target_dir: str = DATA_DIR) -> Dict[str, Any]:
    """
    Download dataset via KaggleHub or verify existing files in target_dir.
    """
    ensure_directories()
    verification = verify_dataset_files(target_dir)
    if verification["valid"]:
        logger.info("All required dataset files are already present in target directory.")
        return {"success": True, "message": "Dataset verified successfully.", "source": "local"}

    logger.info("Attempting KaggleHub download for 'rohanrao/formula-1-world-championship-1950-2020'...")
    try:
        import kagglehub

        download_path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
        logger.info(f"KaggleHub downloaded files to: {download_path}")

        # Copy required CSV files to local data folder
        copied_files = []
        for root, _, files in os.walk(download_path):
            for file in files:
                if file.endswith(".csv"):
                    src = os.path.join(root, file)
                    dst = os.path.join(target_dir, file)
                    shutil.copy2(src, dst)
                    copied_files.append(file)

        verif_after = verify_dataset_files(target_dir)
        if verif_after["valid"]:
            return {
                "success": True,
                "message": f"Successfully downloaded and verified {len(copied_files)} files.",
                "source": "kagglehub",
            }
        else:
            return {
                "success": False,
                "message": f"Downloaded but missing required files: {verif_after['missing_files']}",
                "source": "kagglehub_incomplete",
            }
    except Exception as e:
        logger.error(f"KaggleHub download error: {str(e)}")
        return {
            "success": False,
            "message": f"Kaggle download failed ({str(e)}). Please verify credentials or upload CSVs manually.",
            "source": "failed",
        }


def load_raw_datasets(data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all required raw CSV files into DataFrames.
    """
    verification = verify_dataset_files(data_dir)
    if not verification["valid"]:
        raise FileNotFoundError(
            f"Missing required CSV files in {data_dir}: {verification['missing_files']}"
        )

    dfs = {}
    for filename in REQUIRED_CSV_FILES:
        name = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)
        logger.info(f"Loading {filename} from {filepath}...")
        dfs[name] = pd.read_csv(filepath)
    return dfs


def preprocess_data(
    data_dir: str = DATA_DIR,
    min_year: int = 2018,
    max_year: int = 2024,
    min_lap_ms: int = 55000,
    max_lap_ms: int = 160000,
) -> pd.DataFrame:
    """
    Complete data preprocessing pipeline:
    - Merges lap_times with races, results, drivers, and circuits.
    - Prevents data leakage by exclusively using pre-race and contemporaneous features.
    - Filters out extreme outlier laps (safety cars, pit stops > 160s, red flags).
    - Sorts chronologically by raceId, driverId, and lap.
    """
    dfs = load_raw_datasets(data_dir)

    laps = dfs["lap_times"]
    races = dfs["races"]
    results = dfs["results"]
    drivers = dfs["drivers"]
    circuits = dfs["circuits"]

    # Filter races by year
    races_filtered = races[(races["year"] >= min_year) & (races["year"] <= max_year)].copy()
    races_subset = races_filtered[["raceId", "year", "round", "circuitId", "name", "date"]].rename(
        columns={"name": "race_name"}
    )

    # Merge lap_times with races
    merged = laps.merge(races_subset, on="raceId", how="inner")

    # Merge pre-race results info (grid, constructorId). Note: do NOT merge final position or points!
    results_subset = results[["raceId", "driverId", "grid", "constructorId"]].copy()
    merged = merged.merge(results_subset, on=["raceId", "driverId"], how="left")

    # Merge driver info
    drivers_subset = drivers[["driverId", "driverRef", "code", "forename", "surname", "nationality"]].copy()
    drivers_subset["driver_name"] = drivers_subset["forename"] + " " + drivers_subset["surname"]
    merged = merged.merge(drivers_subset, on="driverId", how="left")

    # Merge circuit info
    circuits_subset = circuits[["circuitId", "circuitRef", "name", "location", "country", "alt"]].rename(
        columns={"name": "circuit_name"}
    )
    merged = merged.merge(circuits_subset, on="circuitId", how="left")

    # Handle missing values
    merged["grid"] = merged["grid"].fillna(10).astype(int)
    merged["constructorId"] = merged["constructorId"].fillna(0).astype(int)
    merged["code"] = merged["code"].fillna("UNK")

    # Filter out extreme lap time outliers (yellow flags, red flags, crash laps)
    original_len = len(merged)
    merged = merged[(merged["milliseconds"] >= min_lap_ms) & (merged["milliseconds"] <= max_lap_ms)].copy()
    filtered_len = len(merged)
    logger.info(
        f"Filtered {original_len - filtered_len} outlier laps outside [{min_lap_ms}, {max_lap_ms}] ms. Remaining laps: {filtered_len}"
    )

    # Sort strictly chronologically
    merged = merged.sort_values(by=["raceId", "driverId", "lap"]).reset_index(drop=True)

    return merged


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = 5,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "milliseconds",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create sequential sliding windows of length `sequence_length`.
    For each (raceId, driverId) group:
      Inputs: [L_{t-4}, L_{t-3}, L_{t-2}, L_{t-1}, L_t] feature vectors
      Target: L_{t+1} milliseconds (the next lap time)
    
    Guarantees:
      - Only consecutive laps without telemetry gaps are windowed.
      - Never leaks future laps or future race positions.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLUMNS

    X_list = []
    y_list = []
    meta_rows = []

    # Group by race and driver
    groups = df.groupby(["raceId", "driverId"])
    logger.info(f"Generating sequences from {len(groups)} driver-race stints...")

    for (race_id, driver_id), group in groups:
        if len(group) <= sequence_length:
            continue

        grp_sorted = group.sort_values("lap").reset_index(drop=True)
        laps = grp_sorted["lap"].values
        feature_vals = grp_sorted[feature_cols].values
        target_vals = grp_sorted[target_col].values

        # Sliding window
        for i in range(len(grp_sorted) - sequence_length):
            window_laps = laps[i : i + sequence_length + 1]
            # Verify laps are strictly consecutive
            is_consecutive = np.all(np.diff(window_laps) == 1)
            if not is_consecutive:
                continue

            X_seq = feature_vals[i : i + sequence_length]
            y_target = target_vals[i + sequence_length]

            X_list.append(X_seq)
            y_list.append(y_target)

            # Metadata for evaluation tracking
            meta_rows.append(
                {
                    "raceId": race_id,
                    "driverId": driver_id,
                    "target_lap": window_laps[-1],
                    "year": grp_sorted.iloc[i + sequence_length]["year"],
                    "race_name": grp_sorted.iloc[i + sequence_length]["race_name"],
                    "driver_name": grp_sorted.iloc[i + sequence_length]["driver_name"],
                    "circuit_name": grp_sorted.iloc[i + sequence_length]["circuit_name"],
                    "actual_time_str": grp_sorted.iloc[i + sequence_length]["time"],
                    "actual_ms": y_target,
                }
            )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    logger.info(f"Generated {len(X)} valid sequence samples. Input shape: {X.shape}, Target shape: {y.shape}")
    return X, y, meta_df


def split_sequences_chronologically(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    test_year: int = 2024,
    val_split_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Split sequences chronologically to prevent temporal data leakage:
      - Train/Val: years < test_year (e.g. 2018-2023)
      - Test: year == test_year (e.g. 2024)
    """
    test_mask = meta_df["year"] == test_year
    train_val_mask = meta_df["year"] < test_year

    if not np.any(test_mask):
        # Fallback if specific year not found: use last 20% of races
        unique_races = meta_df["raceId"].unique()
        split_idx = int(len(unique_races) * 0.8)
        train_val_races = unique_races[:split_idx]
        test_races = unique_races[split_idx:]
        train_val_mask = meta_df["raceId"].isin(train_val_races)
        test_mask = meta_df["raceId"].isin(test_races)

    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    meta_train_val = meta_df[train_val_mask].reset_index(drop=True)

    X_test = X[test_mask]
    y_test = y[test_mask]
    meta_test = meta_df[test_mask].reset_index(drop=True)

    # Validation split from training (chronological by race)
    unique_train_races = meta_train_val["raceId"].unique()
    n_val_races = max(1, int(len(unique_train_races) * val_split_ratio))
    val_races = unique_train_races[-n_val_races:]
    train_races = unique_train_races[:-n_val_races]

    train_mask = meta_train_val["raceId"].isin(train_races)
    val_mask = meta_train_val["raceId"].isin(val_races)

    X_train = X_train_val[train_mask]
    y_train = y_train_val[train_mask]
    meta_train = meta_train_val[train_mask].reset_index(drop=True)

    X_val = X_train_val[val_mask]
    y_val = y_train_val[val_mask]
    meta_val = meta_train_val[val_mask].reset_index(drop=True)

    logger.info(
        f"Chronological Split Summary -> Train: {len(X_train)} samples, "
        f"Val: {len(X_val)} samples, Test: {len(X_test)} samples"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


def fit_and_save_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: str = MODELS_DIR,
) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit StandardScaler on training features and training targets.
    Save fitted scalers to models/ directory.
    """
    ensure_directories()
    n_samples, seq_len, n_features = X_train.shape

    # Reshape 3D to 2D for feature scaler
    X_flat = X_train.reshape(-1, n_features)
    scaler_X = StandardScaler()
    scaler_X.fit(X_flat)

    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    scaler_X_path = os.path.join(models_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(models_dir, "scaler_y.pkl")
    # Also save unified scaler.pkl for compatibility with requirement
    unified_scaler_path = os.path.join(models_dir, "scaler.pkl")

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, unified_scaler_path)

    logger.info(f"Saved scalers to {models_dir}")
    return scaler_X, scaler_y


def load_scalers(models_dir: str = MODELS_DIR) -> Tuple[StandardScaler, StandardScaler]:
    """
    Load scalers from models/ directory.
    """
    scaler_X_path = os.path.join(models_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(models_dir, "scaler_y.pkl")
    unified_scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        return scaler_X, scaler_y
    elif os.path.exists(unified_scaler_path):
        scalers = joblib.load(unified_scaler_path)
        return scalers["scaler_X"], scalers["scaler_y"]
    else:
        raise FileNotFoundError(f"Scalers not found in {models_dir}")


def transform_data(
    X: np.ndarray,
    y: Optional[np.ndarray],
    scaler_X: StandardScaler,
    scaler_y: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scale 3D features and 1D targets using fitted scalers.
    """
    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled_flat = scaler_X.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

    y_scaled = None
    if y is not None and scaler_y is not None:
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled


def inverse_transform_target(
    y_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """
    Inverse scale predicted targets back to raw milliseconds.
    """
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


if __name__ == "__main__":
    ensure_directories()
    print("Verifying dataset files...")
    v = verify_dataset_files()
    print("Verification result:", json.dumps(v, indent=2))
