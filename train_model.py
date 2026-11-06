"""
Formula 1 Lap Time Prediction - Model Training Module
Builds, compiles, and trains the LSTM Deep Learning architecture.
Saves model artifacts, scalers, metadata, and training history.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Tuple

# Set local cache paths before importing TF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_processing import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    DEFAULT_FEATURE_COLUMNS,
    ensure_directories,
    preprocess_data,
    create_sequences,
    split_sequences_chronologically,
    fit_and_save_scalers,
    transform_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_lstm_model(
    sequence_length: int = 5,
    num_features: int = len(DEFAULT_FEATURE_COLUMNS),
    learning_rate: float = 0.001,
) -> Sequential:
    """
    Build LSTM Deep Learning architecture as specified:
      Input Layer (seq_length, num_features)
      ↓
      LSTM (64 units, return_sequences=True)
      ↓
      LSTM (32 units)
      ↓
      Dense (32 units, ReLU)
      ↓
      Dense (1 output)
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features), name="lap_sequence_input"),
        LSTM(64, return_sequences=True, name="lstm_layer_1"),
        LSTM(32, return_sequences=False, name="lstm_layer_2"),
        Dense(32, activation="relu", name="dense_intermediate"),
        Dense(1, activation="linear", name="predicted_lap_time"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    logger.info("Built LSTM architecture successfully:")
    model.summary(print_fn=logger.info)
    return model


def train_lstm_pipeline(
    epochs: int = 15,
    batch_size: int = 128,
    sequence_length: int = 5,
    min_year: int = 2019,
    test_year: int = 2024,
    learning_rate: float = 0.001,
    patience: int = 5,
    models_dir: str = MODELS_DIR,
    outputs_dir: str = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """
    Complete end-to-end training pipeline:
      1. Preprocess raw data with temporal filtering.
      2. Generate consecutive sliding window sequences.
      3. Chronologically split Train (<=2023) and Test (2024).
      4. Fit feature & target StandardScaler.
      5. Train LSTM with EarlyStopping.
      6. Save model, scalers, metadata, and history.
    """
    ensure_directories()
    t_start = time.time()

    logger.info(f"Step 1: Preprocessing F1 dataset (years {min_year}-{test_year})...")
    df = preprocess_data(
        data_dir=DATA_DIR,
        min_year=min_year,
        max_year=test_year,
        min_lap_ms=55000,
        max_lap_ms=160000,
    )

    logger.info("Step 2: Creating consecutive sequences...")
    X, y, meta_df = create_sequences(
        df=df,
        sequence_length=sequence_length,
        feature_cols=DEFAULT_FEATURE_COLUMNS,
        target_col="milliseconds",
    )

    logger.info("Step 3: Performing chronological train/val/test split...")
    split_data = split_sequences_chronologically(
        X=X,
        y=y,
        meta_df=meta_df,
        test_year=test_year,
        val_split_ratio=0.15,
    )

    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    meta_test = split_data["meta_test"]

    logger.info("Step 4: Fitting and saving standard scalers...")
    scaler_X, scaler_y = fit_and_save_scalers(X_train, y_train, models_dir=models_dir)

    logger.info("Scaling features and targets...")
    X_train_s, y_train_s = transform_data(X_train, y_train, scaler_X, scaler_y)
    X_val_s, y_val_s = transform_data(X_val, y_val, scaler_X, scaler_y)
    X_test_s, y_test_s = transform_data(X_test, y_test, scaler_X, scaler_y)

    # Save feature metadata
    metadata = {
        "sequence_length": sequence_length,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "num_features": len(DEFAULT_FEATURE_COLUMNS),
        "target_column": "milliseconds",
        "train_years": f"{min_year}-{test_year - 1}",
        "test_year": test_year,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
        json.dump(DEFAULT_FEATURE_COLUMNS, f, indent=2)
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Step 5: Building LSTM model...")
    model = build_lstm_model(
        sequence_length=sequence_length,
        num_features=len(DEFAULT_FEATURE_COLUMNS),
        learning_rate=learning_rate,
    )

    model_path = os.path.join(models_dir, "lstm_model.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    logger.info(f"Step 6: Training LSTM model for up to {epochs} epochs (batch size: {batch_size})...")
    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    training_duration = time.time() - t_start
    logger.info(f"Model training finished in {training_duration:.2f} seconds.")

    # Save model explicitly
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_dict = {
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "mae": [float(x) for x in history.history.get("mae", [])],
        "val_mae": [float(x) for x in history.history.get("val_mae", [])],
        "lr": [float(x) for x in history.history.get("lr", [learning_rate])],
        "epochs_trained": len(history.history.get("loss", [])),
        "duration_seconds": round(training_duration, 2),
    }
    history_path = os.path.join(outputs_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Save test dataset arrays for evaluation module
    np.save(os.path.join(outputs_dir, "X_test.npy"), X_test)
    np.save(os.path.join(outputs_dir, "y_test.npy"), y_test)
    meta_test.to_csv(os.path.join(outputs_dir, "meta_test.csv"), index=False)

    return {
        "model": model,
        "history": history_dict,
        "metadata": metadata,
        "model_path": model_path,
        "duration_seconds": training_duration,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model for F1 lap time prediction.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--seq-len", type=int, default=5, help="Number of previous laps in sequence")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--min-year", type=int, default=2019, help="Earliest year in dataset")
    parser.add_argument("--test-year", type=int, default=2024, help="Test year for chronological split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("FORMULA 1 LAP TIME PREDICTION - LSTM MODEL TRAINING")
    print("=" * 60)
    result = train_lstm_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        min_year=args.min_year,
        test_year=args.test_year,
        learning_rate=args.lr,
    )
    print("=" * 60)
    print(f"TRAINING COMPLETE in {result['duration_seconds']:.2f}s!")
    print(f"Model saved to: {result['model_path']}")
    print("=" * 60)
