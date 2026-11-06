"""
🏎️ F1 LAP TIME PREDICTION SYSTEM
AI-Powered Formula 1 Lap Time Prediction Using LSTM Deep Learning
Final Year BTech Project
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional

# Set local cache paths before importing TF / Matplotlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_processing import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    format_ms,
    time_to_ms,
    verify_dataset_files,
    download_kaggle_dataset,
    load_raw_datasets,
)
from prediction import (
    predict_next_lap,
    predict_from_historical_race,
    get_driver_list,
    get_race_list,
    get_cached_metadata,
)
from evaluation import evaluate_model_pipeline
from train_model import train_lstm_pipeline

# Configure Streamlit page
st.set_page_config(
    page_title="F1 Lap Time Prediction | Deep Learning LSTM",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_file = os.path.join(BASE_DIR, "assets", "style.css")
if os.path.exists(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Helper functions to load metadata and state
@st.cache_data
def get_dataset_overview():
    """Load dataset verification and summary statistics."""
    v = verify_dataset_files()
    if not v["valid"]:
        return {"valid": False, "summary": {}, "verification": v}

    try:
        raw_data = load_raw_datasets()
        laps = raw_data["lap_times"]
        races = raw_data["races"]
        drivers = raw_data["drivers"]
        circuits = raw_data["circuits"]
        results = raw_data["results"]

        summary = {
            "total_races": len(races),
            "total_drivers": len(drivers),
            "total_circuits": len(circuits),
            "total_laps": len(laps),
            "min_year": int(races["year"].min()),
            "max_year": int(races["year"].max()),
            "avg_lap_ms": float(laps["milliseconds"].mean()),
        }
        return {"valid": True, "summary": summary, "verification": v}
    except Exception as e:
        return {"valid": False, "error": str(e), "verification": v}


def get_model_status():
    """Check if trained model and evaluation outputs exist."""
    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    history_path = os.path.join(OUTPUTS_DIR, "training_history.json")
    eval_path = os.path.join(OUTPUTS_DIR, "evaluation_results.json")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    has_model = os.path.exists(model_path)
    eval_metrics = None
    training_history = None
    meta_info = None

    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                eval_metrics = json.load(f)
        except Exception:
            pass

    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                training_history = json.load(f)
        except Exception:
            pass

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta_info = json.load(f)
        except Exception:
            pass

    return {
        "has_model": has_model,
        "eval_metrics": eval_metrics,
        "training_history": training_history,
        "meta_info": meta_info,
    }


# Sidebar Navigation & System Controls
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h2 style="color: #E10600; margin: 0; font-weight: 900; letter-spacing: 2px;">🏎️ F1 PREDICTOR</h2>
        <p style="color: #8F9CAE; font-size: 0.85rem; margin-top: 4px;">Deep Learning LSTM Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "📊 Data Analysis & EDA",
        "⏱️ Lap Time Prediction",
        "📈 Model Performance",
        "ℹ️ About Project",
    ],
    index=0,
)

st.sidebar.markdown("---")

# Model Status Widget in Sidebar
m_status = get_model_status()
if m_status["has_model"]:
    st.sidebar.markdown(
        """
        <div style="background: #161A23; border: 1px solid #00D26A; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="color: #00D26A; font-weight: 700; font-size: 0.85rem;">● MODEL STATUS: ACTIVE</div>
            <div style="color: #E0E4EC; font-size: 0.8rem; margin-top: 4px;">Trained LSTM Ready (31,937 params)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div style="background: #161A23; border: 1px solid #FF5A55; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="color: #FF5A55; font-weight: 700; font-size: 0.85rem;">● MODEL STATUS: NOT TRAINED</div>
            <div style="color: #E0E4EC; font-size: 0.8rem; margin-top: 4px;">Train model below to activate prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Quick Retrain Collapsible in Sidebar
with st.sidebar.expander("⚙️ Model Training Controls", expanded=False):
    st.markdown("#### Hyperparameters")
    epochs_in = st.slider("Epochs", min_value=5, max_value=30, value=15, step=1)
    batch_in = st.select_slider("Batch Size", options=[64, 128, 256, 512], value=128)
    seq_in = st.slider("Sequence Length", min_value=3, max_value=10, value=5, step=1)

    if st.button("🚀 Train / Retrain Model", use_container_width=True):
        with st.spinner("Training LSTM model on F1 dataset... This takes ~1 minute"):
            try:
                res = train_lstm_pipeline(
                    epochs=epochs_in,
                    batch_size=batch_in,
                    sequence_length=seq_in,
                    min_year=2019,
                    test_year=2024,
                )
                eval_res = evaluate_model_pipeline()
                st.success(f"Model trained successfully in {res['duration_seconds']:.1f}s!")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

st.sidebar.markdown(
    """
    <div style="text-align: center; color: #5A6578; font-size: 0.75rem; margin-top: 30px;">
        Final Year BTech Project<br>Formula 1 Telemetry Analysis & Deep Learning
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if nav_page == "🏠 Home Dashboard":
    st.markdown(
        """
        <div class="f1-hero">
            <h1 class="f1-title">🏎️ <span>F1 LAP TIME</span> PREDICTION SYSTEM</h1>
            <div class="f1-subtitle">AI-Powered Formula 1 Lap Time Forecasting Using LSTM Recurrent Neural Networks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview = get_dataset_overview()

    # Metric Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    if overview["valid"]:
        s = overview["summary"]
        with col1:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">Historical Grands Prix</div>
                    <div class="f1-metric-value">{s['total_races']:,}</div>
                    <div class="f1-metric-delta delta-pos">Seasons {s['min_year']}–{s['max_year']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">F1 Drivers</div>
                    <div class="f1-metric-value">{s['total_drivers']:,}</div>
                    <div class="f1-metric-delta delta-pos">World Champions & Rookies</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">Total Laps Logged</div>
                    <div class="f1-metric-value">{s['total_laps']:,}</div>
                    <div class="f1-metric-delta delta-pos">Telemetry Milestones</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">Racing Circuits</div>
                    <div class="f1-metric-value">{s['total_circuits']}</div>
                    <div class="f1-metric-delta delta-pos">Worldwide Tracks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Performance Highlights Row
    st.markdown("### 🏆 Model Performance Highlights")
    if m_status["eval_metrics"]:
        em = m_status["eval_metrics"]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">Mean Absolute Error</div>
                    <div class="f1-metric-value" style="color: #00D26A;">{em['mae_sec']:.3f} s</div>
                    <div class="f1-metric-delta delta-pos">{em['mae_ms']:.1f} ms avg error</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol2:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">Root Mean Sq. Error</div>
                    <div class="f1-metric-value">{em['rmse_sec']:.3f} s</div>
                    <div class="f1-metric-delta delta-pos">{em['rmse_ms']:.1f} ms RMSE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol3:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">R² Goodness of Fit</div>
                    <div class="f1-metric-value" style="color: #FFB800;">{em['r2_score']:.4f}</div>
                    <div class="f1-metric-delta delta-pos">{em['r2_score']*100:.1f}% variance explained</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol4:
            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="f1-metric-label">Mean Relative Error</div>
                    <div class="f1-metric-value" style="color: #00D26A;">{em['mean_absolute_percentage_error']:.2f}%</div>
                    <div class="f1-metric-delta delta-pos">Across {em['total_test_samples']:,} test laps</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Model evaluation results not found. Click 'Train / Retrain Model' in the sidebar.")

    st.markdown("<br>", unsafe_allow_html=True)

    # System Status & Architecture Overview Columns
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### 🗄️ Dataset Health & Verification")
        v = overview["verification"]
        for fname, info in v["files"].items():
            icon = "✅" if info["exists"] else "❌"
            badge = f'<span class="f1-badge badge-ready">{info["size_mb"]} MB</span>' if info["exists"] else '<span class="f1-badge badge-red">Missing</span>'
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #161A23; border: 1px solid #2A303F; border-radius: 6px; margin-bottom: 8px;">
                    <div><span style="margin-right: 8px;">{icon}</span> <code>{fname}</code></div>
                    <div>{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("### 🧠 Deep Learning Architecture Spec")
        st.markdown(
            """
            <div style="background: #161A23; border: 1px solid #2A303F; border-radius: 8px; padding: 18px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                    <tr style="border-bottom: 1px solid #2A303F;">
                        <td style="padding: 8px 0; color: #8F9CAE;">Architecture:</td>
                        <td style="padding: 8px 0; font-weight: 700; color: #FFFFFF;">Stacked LSTM Network</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2A303F;">
                        <td style="padding: 8px 0; color: #8F9CAE;">Input Sequence:</td>
                        <td style="padding: 8px 0; font-family: 'JetBrains Mono', monospace; color: #FF5A55;">(None, 5, 7) — 5 Previous Laps</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2A303F;">
                        <td style="padding: 8px 0; color: #8F9CAE;">Layer 1:</td>
                        <td style="padding: 8px 0; color: #FFFFFF;">LSTM (64 units, return_sequences=True)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2A303F;">
                        <td style="padding: 8px 0; color: #8F9CAE;">Layer 2:</td>
                        <td style="padding: 8px 0; color: #FFFFFF;">LSTM (32 units)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2A303F;">
                        <td style="padding: 8px 0; color: #8F9CAE;">Intermediate Layer:</td>
                        <td style="padding: 8px 0; color: #FFFFFF;">Dense (32 units, ReLU activation)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #2A303F;">
                        <td style="padding: 8px 0; color: #8F9CAE;">Output:</td>
                        <td style="padding: 8px 0; color: #00D26A; font-weight: 700;">Predicted Next Lap Time (ms)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #8F9CAE;">Optimization:</td>
                        <td style="padding: 8px 0; color: #FFFFFF;">Adam (lr=0.001), Loss=MSE, Metric=MAE</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: DATA ANALYSIS & EDA
# ==============================================================================
elif nav_page == "📊 Data Analysis & EDA":
    st.markdown(
        """
        <div class="f1-hero">
            <h1 class="f1-title">📊 <span>HISTORICAL DATA</span> ANALYSIS & EDA</h1>
            <div class="f1-subtitle">Exploratory Data Analysis across F1 Seasons, Circuits, and Drivers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = get_cached_metadata()
    races_df = data["races"]
    drivers_df = data["drivers"]
    circuits_df = data["circuits"]
    laps_df = data["lap_times"]
    results_df = data["results"]

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    recent_years = sorted(races_df[races_df["year"] >= 2018]["year"].unique(), reverse=True)
    with f_col1:
        sel_year = st.selectbox("Select F1 Championship Season", options=recent_years, index=0)

    season_races = races_df[races_df["year"] == sel_year].sort_values("round")
    race_options = {r["name"]: r["raceId"] for _, r in season_races.iterrows()}
    with f_col2:
        sel_race_name = st.selectbox("Select Grand Prix", options=list(race_options.keys()))
        sel_race_id = race_options[sel_race_name]

    race_laps = laps_df[laps_df["raceId"] == sel_race_id]
    race_driver_ids = race_laps["driverId"].unique()
    race_drivers = drivers_df[drivers_df["driverId"].isin(race_driver_ids)].copy()
    race_drivers["name"] = race_drivers["forename"] + " " + race_drivers["surname"]
    driver_options = {d["name"]: d["driverId"] for _, d in race_drivers.iterrows()}

    with f_col3:
        sel_driver_names = st.multiselect(
            "Select Drivers to Compare",
            options=list(driver_options.keys()),
            default=list(driver_options.keys())[:4] if len(driver_options) >= 4 else list(driver_options.keys()),
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🏎️ Race Pace Progression", "📦 Driver Pace Distribution", "🌍 Circuit Comparisons"])

    with tab1:
        st.markdown(f"#### Lap Time Progression: {sel_race_name} ({sel_year})")
        if sel_driver_names:
            selected_ids = [driver_options[name] for name in sel_driver_names]
            filtered_laps = race_laps[race_laps["driverId"].isin(selected_ids)].copy()
            filtered_laps = filtered_laps.merge(race_drivers[["driverId", "name"]], on="driverId")
            filtered_laps["seconds"] = filtered_laps["milliseconds"] / 1000.0

            # Filter out pit stop outliers > 140s for clean progression visual
            filtered_laps_clean = filtered_laps[filtered_laps["seconds"] < 130].copy()

            fig = px.line(
                filtered_laps_clean,
                x="lap",
                y="seconds",
                color="name",
                markers=True,
                title=f"Lap Time Trajectory by Driver — {sel_race_name}",
                labels={"lap": "Lap Number", "seconds": "Lap Time (seconds)", "name": "Driver"},
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="#161A23",
                paper_bgcolor="#161A23",
                font_family="Titillium Web",
                hovermode="x unified",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one driver to plot race progression.")

    with tab2:
        st.markdown(f"#### Lap Time Variability & Pace Boxplots ({sel_race_name})")
        if sel_driver_names:
            selected_ids = [driver_options[name] for name in sel_driver_names]
            filtered_laps = race_laps[race_laps["driverId"].isin(selected_ids)].copy()
            filtered_laps = filtered_laps.merge(race_drivers[["driverId", "name"]], on="driverId")
            filtered_laps["seconds"] = filtered_laps["milliseconds"] / 1000.0
            filtered_laps_clean = filtered_laps[filtered_laps["seconds"] < 130].copy()

            fig_box = px.box(
                filtered_laps_clean,
                x="name",
                y="seconds",
                color="name",
                points="outliers",
                title="Pace Consistency and Outlier Distribution (Boxplot)",
                labels={"name": "Driver", "seconds": "Lap Time (seconds)"},
                template="plotly_dark",
            )
            fig_box.update_layout(
                plot_bgcolor="#161A23",
                paper_bgcolor="#161A23",
                font_family="Titillium Web",
                height=450,
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.markdown("#### Circuit Pace Comparison (Modern F1 Era)")
        merged_c = laps_df.merge(races_df[races_df["year"] >= 2021][["raceId", "circuitId"]], on="raceId")
        merged_c = merged_c.merge(circuits_df[["circuitId", "name", "country"]], on="circuitId")
        merged_c = merged_c[(merged_c["milliseconds"] >= 55000) & (merged_c["milliseconds"] <= 140000)]

        circuit_stats = merged_c.groupby(["name", "country"]).agg(
            avg_ms=("milliseconds", "mean"),
            min_ms=("milliseconds", "min"),
            total_laps=("lap", "count")
        ).reset_index()
        circuit_stats["avg_sec"] = circuit_stats["avg_ms"] / 1000.0
        circuit_stats["min_sec"] = circuit_stats["min_ms"] / 1000.0
        circuit_stats = circuit_stats.sort_values("avg_sec").head(15)

        fig_circ = px.bar(
            circuit_stats,
            x="avg_sec",
            y="name",
            orientation="h",
            color="avg_sec",
            color_continuous_scale="Reds",
            title="Fastest F1 Circuits by Average Lap Time (2021–2024)",
            labels={"avg_sec": "Average Lap Time (seconds)", "name": "Circuit Name"},
            template="plotly_dark",
        )
        fig_circ.update_layout(
            plot_bgcolor="#161A23",
            paper_bgcolor="#161A23",
            font_family="Titillium Web",
            height=500,
            yaxis={"categoryorder": "total descending"},
        )
        st.plotly_chart(fig_circ, use_container_width=True)


# ==============================================================================
# PAGE 3: LAP TIME PREDICTION
# ==============================================================================
elif nav_page == "⏱️ Lap Time Prediction":
    st.markdown(
        """
        <div class="f1-hero">
            <h1 class="f1-title">⏱️ <span>INTERACTIVE LAP TIME</span> PREDICTOR</h1>
            <div class="f1-subtitle">Real-time LSTM Neural Network Inference for Next Lap Lap-Time Forecasting</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not m_status["has_model"]:
        st.error("⚠️ Trained model not detected! Please use 'Train / Retrain Model' in the sidebar to build the model first.")
        st.stop()

    pred_mode = st.radio(
        "Select Prediction Mode",
        ["🏁 Historical Race Telemetry Validation (Real Race Replay)", "🎮 Custom Scenario Simulator (Sandbox)"],
        horizontal=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE A: HISTORICAL RACE REPLAY
    # -------------------------------------------------------------------------
    if "Historical" in pred_mode:
        st.markdown("### 🏁 Historical Race Validation")
        st.info("Select a Grand Prix, driver, and lap number to pull the preceding 5 consecutive laps from historical telemetry, feed them to the trained LSTM model, and compare the prediction with ground truth.")

        col_a, col_b, col_c = st.columns(3)
        data = get_cached_metadata()
        races_df = data["races"]
        drivers_df = data["drivers"]
        laps_df = data["lap_times"]

        recent_races = races_df[races_df["year"] >= 2022].sort_values(by=["year", "round"], ascending=[False, True])
        race_dict = {f"{r['year']} {r['name']}": r["raceId"] for _, r in recent_races.iterrows()}

        with col_a:
            selected_race_label = st.selectbox("Select Grand Prix", options=list(race_dict.keys()), index=0)
            sel_race_id = race_dict[selected_race_label]

        # Get drivers who competed in this race
        race_laps = laps_df[laps_df["raceId"] == sel_race_id]
        race_driver_ids = race_laps["driverId"].unique()
        race_drivers = drivers_df[drivers_df["driverId"].isin(race_driver_ids)].copy()
        race_drivers["name"] = race_drivers["forename"] + " " + race_drivers["surname"]
        driver_dict = {d["name"]: d["driverId"] for _, d in race_drivers.iterrows()}

        with col_b:
            selected_driver_name = st.selectbox("Select Driver", options=list(driver_dict.keys()))
            sel_driver_id = driver_dict[selected_driver_name]

        # Get lap range for driver
        driver_laps = race_laps[race_laps["driverId"] == sel_driver_id].sort_values("lap")
        max_lap = int(driver_laps["lap"].max()) if not driver_laps.empty else 10

        with col_c:
            selected_lap = st.slider("Target Lap to Predict", min_value=6, max_value=max(7, max_lap), value=min(25, max_lap))

        if st.button("🔮 Run LSTM Model Prediction", key="btn_hist_pred"):
            with st.spinner("Executing LSTM Deep Learning inference..."):
                try:
                    res = predict_from_historical_race(
                        race_id=sel_race_id,
                        driver_id=sel_driver_id,
                        target_lap=selected_lap,
                    )

                    # Display Prediction Cards
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.markdown(
                            f"""
                            <div class="f1-lap-display">
                                <div class="f1-lap-label">Predicted Lap {res['predicted_lap']}</div>
                                <div class="f1-lap-time-large">{res['predicted_formatted']}</div>
                                <div style="color: #8F9CAE; font-size: 0.9rem;">{res['predicted_ms']:,} ms</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with res_col2:
                        st.markdown(
                            f"""
                            <div class="f1-lap-display" style="border-color: #3880FF;">
                                <div class="f1-lap-label">Actual Ground Truth</div>
                                <div class="f1-lap-time-large" style="text-shadow: 0 0 15px rgba(56, 128, 255, 0.5);">{res['actual_formatted']}</div>
                                <div style="color: #8F9CAE; font-size: 0.9rem;">{res['actual_ms']:,} ms</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with res_col3:
                        delta_color = "#00D26A" if abs(res['error_seconds']) < 1.0 else "#FFB800"
                        st.markdown(
                            f"""
                            <div class="f1-lap-display" style="border-color: {delta_color};">
                                <div class="f1-lap-label">Prediction Delta / Error</div>
                                <div class="f1-lap-time-large" style="color: {delta_color}; text-shadow: 0 0 15px {delta_color};">{res['error_seconds']:+.3f} s</div>
                                <div style="color: #8F9CAE; font-size: 0.9rem;">Model Accuracy: {res['accuracy_percent']:.2f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Interactive Plotly chart showing sequence, prediction, and actual
                    st.markdown("#### 📈 Preceding Sequence & Prediction Trajectory")
                    laps_x = list(range(selected_lap - 5, selected_lap))
                    times_y = [x / 1000.0 for x in res["input_sequence_ms"]]

                    fig = go.Figure()

                    # Preceding laps trace
                    fig.add_trace(go.Scatter(
                        x=laps_x,
                        y=times_y,
                        mode="lines+markers",
                        name="Preceding Laps (Input)",
                        line=dict(color="#E10600", width=3),
                        marker=dict(size=8, color="#FFFFFF"),
                    ))

                    # Actual target lap trace
                    fig.add_trace(go.Scatter(
                        x=[laps_x[-1], selected_lap],
                        y=[times_y[-1], res["actual_seconds"]],
                        mode="lines+markers",
                        name="Actual Lap Time",
                        line=dict(color="#3880FF", width=3, dash="dot"),
                        marker=dict(size=12, symbol="circle", color="#3880FF"),
                    ))

                    # Predicted target lap trace
                    fig.add_trace(go.Scatter(
                        x=[laps_x[-1], selected_lap],
                        y=[times_y[-1], res["predicted_seconds"]],
                        mode="lines+markers",
                        name="LSTM Predicted Lap Time",
                        line=dict(color="#00D26A", width=3, dash="dash"),
                        marker=dict(size=12, symbol="star", color="#00D26A"),
                    ))

                    fig.update_layout(
                        title=f"{res['driver_name']} — Laps {selected_lap-5} to {selected_lap} ({res['race_name']})",
                        xaxis_title="Lap Number",
                        yaxis_title="Lap Time (seconds)",
                        plot_bgcolor="#161A23",
                        paper_bgcolor="#161A23",
                        font_family="Titillium Web",
                        template="plotly_dark",
                        height=420,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    # -------------------------------------------------------------------------
    # MODE B: CUSTOM SCENARIO SIMULATOR
    # -------------------------------------------------------------------------
    else:
        st.markdown("### 🎮 Custom Lap Scenario Simulator")
        st.info("Input any hypothetical sequence of 5 previous lap times, track position, and circuit to simulate what the LSTM predicts for the next lap.")

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            st.markdown("#### Race Parameters")
            sim_lap = st.number_input("Current Lap Number", min_value=5, max_value=80, value=25)
            sim_pos = st.slider("Current Track Position", min_value=1, max_value=20, value=3)
            sim_grid = st.slider("Starting Grid Position", min_value=1, max_value=20, value=2)

        with sim_col2:
            st.markdown("#### Circuit & Pace Baseline")
            circuit_presets = {
                "Red Bull Ring (Austria - Fast ~66s)": 66000,
                "Silverstone (UK - High Speed ~90s)": 90000,
                "Monza (Italy - Temple of Speed ~83s)": 83000,
                "Spa-Francorchamps (Belgium ~106s)": 106000,
                "Monaco GP (Tight Street Track ~75s)": 75000,
            }
            sel_preset = st.selectbox("Pace Preset", options=list(circuit_presets.keys()))
            base_pace = circuit_presets[sel_preset]

        st.markdown("#### ⏱️ Input 5 Consecutive Previous Laps (m:ss.sss)")
        lap_cols = st.columns(5)
        input_laps_ms = []

        # Generate realistic default sequence with slight tire wear
        default_offsets = [0, 150, 320, 280, 500]
        for i, col in enumerate(lap_cols):
            with col:
                lap_num = sim_lap - 4 + i
                def_val = format_ms(base_pace + default_offsets[i])
                val_str = st.text_input(f"Lap {lap_num}", value=def_val, key=f"custom_lap_{i}")
                ms_val = time_to_ms(val_str)
                if ms_val is None:
                    ms_val = base_pace
                input_laps_ms.append(ms_val)

        if st.button("🚀 Simulate Next Lap (Lap " + str(sim_lap + 1) + ")", key="btn_sim_pred"):
            with st.spinner("Running deep learning simulation..."):
                try:
                    pred_res = predict_next_lap(
                        recent_laps_ms=input_laps_ms,
                        current_lap=sim_lap,
                        position=sim_pos,
                        circuit_id=1,
                        driver_id=1,
                        constructor_id=1,
                        grid=sim_grid,
                    )

                    c_res1, c_res2 = st.columns(2)
                    with c_res1:
                        st.markdown(
                            f"""
                            <div class="f1-lap-display">
                                <div class="f1-lap-label">Predicted Next Lap ({pred_res['predicted_lap']})</div>
                                <div class="f1-lap-time-large">{pred_res['predicted_formatted']}</div>
                                <div style="color: #8F9CAE; font-size: 0.9rem;">{pred_res['predicted_ms']:,} ms ({pred_res['predicted_seconds']} s)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with c_res2:
                        trend_text = "Tire Wear / Slowing" if pred_res['delta_vs_previous_sec'] > 0 else "Pace Acceleration"
                        trend_color = "#FF5A55" if pred_res['delta_vs_previous_sec'] > 0 else "#00D26A"
                        st.markdown(
                            f"""
                            <div class="f1-lap-display" style="border-color: {trend_color};">
                                <div class="f1-lap-label">Delta vs Previous Lap</div>
                                <div class="f1-lap-time-large" style="color: {trend_color};">{pred_res['delta_vs_previous_sec']:+.3f} s</div>
                                <div style="color: #8F9CAE; font-size: 0.9rem;">Behavior: {trend_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE & EVALUATION
# ==============================================================================
elif nav_page == "📈 Model Performance":
    st.markdown(
        """
        <div class="f1-hero">
            <h1 class="f1-title">📈 <span>MODEL PERFORMANCE</span> & EVALUATION</h1>
            <div class="f1-subtitle">Validation Loss, Residual Analysis, Error Distributions, and Test Metrics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not m_status["eval_metrics"]:
        st.warning("No evaluation metrics recorded yet. Please run training or evaluation.")
        st.stop()

    em = m_status["eval_metrics"]
    hist = m_status["training_history"]

    # Metrics Summary Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="f1-metric-card">
                <div class="f1-metric-label">MAE</div>
                <div class="f1-metric-value" style="color: #00D26A;">{em['mae_sec']:.3f}s</div>
                <div class="f1-metric-delta delta-pos">{em['mae_ms']:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="f1-metric-card">
                <div class="f1-metric-label">RMSE</div>
                <div class="f1-metric-value">{em['rmse_sec']:.3f}s</div>
                <div class="f1-metric-delta delta-pos">{em['rmse_ms']:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="f1-metric-card">
                <div class="f1-metric-label">MSE</div>
                <div class="f1-metric-value" style="font-size: 1.4rem;">{em['mse_ms']:,.0f}</div>
                <div class="f1-metric-delta delta-pos">Squared ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="f1-metric-card">
                <div class="f1-metric-label">R² Score</div>
                <div class="f1-metric-value" style="color: #FFB800;">{em['r2_score']:.4f}</div>
                <div class="f1-metric-delta delta-pos">{em['r2_score']*100:.1f}% fit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="f1-metric-card">
                <div class="f1-metric-label">Test Samples</div>
                <div class="f1-metric-value">{em['total_test_samples']:,}</div>
                <div class="f1-metric-delta delta-pos">2024 Season</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1: Training History & Residuals
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("#### 📉 Training Loss vs Validation Loss Curve")
        if hist and "loss" in hist:
            epochs_range = list(range(1, len(hist["loss"]) + 1))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=epochs_range, y=hist["loss"], mode="lines+markers", name="Training Loss (MSE)", line=dict(color="#FF5A55", width=2)
            ))
            fig_hist.add_trace(go.Scatter(
                x=epochs_range, y=hist["val_loss"], mode="lines+markers", name="Validation Loss (MSE)", line=dict(color="#00D26A", width=2)
            ))
            fig_hist.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Loss (Mean Squared Error)",
                plot_bgcolor="#161A23",
                paper_bgcolor="#161A23",
                font_family="Titillium Web",
                template="plotly_dark",
                height=380,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with g_col2:
        st.markdown("#### 🎯 Prediction Error Distribution (Residuals)")
        pred_csv = os.path.join(OUTPUTS_DIR, "test_predictions.csv")
        if os.path.exists(pred_csv):
            pred_df = pd.read_csv(pred_csv)
            # Filter within 10s for clean histogram
            res_clean = pred_df[(pred_df["error_sec"] >= -8) & (pred_df["error_sec"] <= 8)]

            fig_res = px.histogram(
                res_clean,
                x="error_sec",
                nbins=60,
                title="Prediction Residuals (Actual - Predicted in Seconds)",
                labels={"error_sec": "Residual Error (seconds)"},
                color_discrete_sequence=["#E10600"],
                template="plotly_dark",
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#00D26A", annotation_text="Ideal Zero Error")
            fig_res.update_layout(
                plot_bgcolor="#161A23",
                paper_bgcolor="#161A23",
                font_family="Titillium Web",
                height=380,
            )
            st.plotly_chart(fig_res, use_container_width=True)

    # Charts Row 2: Actual vs Predicted Scatter
    st.markdown("#### 🔬 Actual vs Predicted Lap Times Correlation (Test Set)")
    if os.path.exists(pred_csv):
        pred_df = pd.read_csv(pred_csv)
        sample_subset = pred_df.sample(min(2000, len(pred_df)), random_state=42)
        sample_subset["actual_sec"] = sample_subset["actual_ms"] / 1000.0
        sample_subset["pred_sec"] = sample_subset["predicted_ms"] / 1000.0

        fig_corr = px.scatter(
            sample_subset,
            x="actual_sec",
            y="pred_sec",
            color="abs_error_sec",
            color_continuous_scale="Viridis",
            labels={"actual_sec": "Actual Lap Time (s)", "pred_sec": "Predicted Lap Time (s)", "abs_error_sec": "Abs Error (s)"},
            title="Actual vs Predicted Lap Time (2,000 Random Samples from 2024 Season)",
            template="plotly_dark",
            hover_data=["driver_name", "race_name", "target_lap"],
        )
        min_v = float(sample_subset["actual_sec"].min())
        max_v = float(sample_subset["actual_sec"].max())
        fig_corr.add_trace(go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Ideal Identity (y = x)",
            line=dict(color="#FF5A55", dash="dash", width=2),
        ))
        fig_corr.update_layout(
            plot_bgcolor="#161A23",
            paper_bgcolor="#161A23",
            font_family="Titillium Web",
            height=480,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Interactive Table
        st.markdown("#### 📋 Test Predictions Sample Inspector")
        st.dataframe(
            pred_df[["race_name", "driver_name", "target_lap", "actual_formatted", "predicted_formatted", "error_sec", "abs_error_sec"]].head(100),
            use_container_width=True,
            height=300,
        )


# ==============================================================================
# PAGE 5: ABOUT PROJECT
# ==============================================================================
elif nav_page == "ℹ️ About Project":
    st.markdown(
        """
        <div class="f1-hero">
            <h1 class="f1-title">ℹ️ <span>ABOUT</span> THIS PROJECT</h1>
            <div class="f1-subtitle">BTech Final Year Capstone Project — Formula 1 Telemetry & Deep Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["📌 Project Overview", "🧠 LSTM Deep Learning Math", "🛠️ Technologies", "🚀 Future Roadmap"])

    with t1:
        st.markdown(
            """
            ### 1. Problem Statement
            Formula 1 is the pinnacle of motorsport where lap-time advantages are measured in thousandths of a second (milliseconds). 
            Predicting lap times accurately during a race is critical for pit-stop strategists, undercut/overcut tactics, and tire life management.
            However, lap-time dynamics are governed by non-linear factors:
            - **Tire Degradation**: Grip degrades continuously per lap, increasing lap times non-linearly.
            - **Fuel Burn-off**: Cars burn ~1.7 kg of fuel per lap, becoming lighter and gaining ~0.08–0.12 seconds per lap.
            - **Track Evolution**: As rubber is laid down on the racing line, track grip improves.
            - **Traffic & Outliers**: Dirty air, defending, and overtaking create fluctuations.

            ### 2. Project Objectives
            - Ingest and clean historical Formula 1 race and lap time telemetry from 1950 to 2024.
            - Formulate lap-time prediction as a strictly causal temporal sequence problem without data leakage.
            - Build, train, and validate a multi-layer Long Short-Term Memory (LSTM) deep recurrent neural network.
            - Achieve an average lap-time forecasting error under 2.5 seconds on modern hybrid era regulations.
            - Deploy a Formula 1-inspired, fully interactive dashboard with real-time scenario simulation and historical telemetry validation.
            """
        )

    with t2:
        st.markdown(
            """
            ### Mathematical Formulation of the LSTM Architecture
            Recurrent Neural Networks (RNNs) suffer from vanishing gradients over long sequences. Long Short-Term Memory (LSTM) cells solve this via a gating mechanism:

            #### 1. Forget Gate
            Decides what information from previous cell state $C_{t-1}$ to discard:
            $$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$$

            #### 2. Input Gate & Candidate State
            Determines what new telemetry features to store:
            $$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$$
            $$\\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)$$

            #### 3. Cell State Update
            Combines forgotten past state and candidate updates:
            $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$

            #### 4. Output Gate & Hidden State
            Outputs the updated representation for the dense regression layers:
            $$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$$
            $$h_t = o_t \\odot \\tanh(C_t)$$

            #### 5. Network Architecture
            ```
            Input Layer: (Batch, Sequence=5, Features=7)
            ↓
            LSTM Layer 1: 64 Units (return_sequences=True) [18,432 params]
            ↓
            LSTM Layer 2: 32 Units (return_sequences=False) [12,416 params]
            ↓
            Dense Layer: 32 Units, ReLU Activation [1,056 params]
            ↓
            Output Layer: 1 Unit, Linear Activation [33 params]
            Total Trainable Parameters: 31,937
            ```
            """
        )

    with t3:
        st.markdown(
            """
            ### Technical Stack & Libraries
            - **Deep Learning Framework**: TensorFlow 2.20 & Keras 3.13
            - **Frontend Dashboard**: Streamlit 1.42 with custom Formula 1 CSS
            - **Data Processing**: Pandas, NumPy, Scikit-learn
            - **Data Visualizations**: Plotly Express & Plotly Graph Objects
            - **Dataset Ingestion**: KaggleHub (`rohanrao/formula-1-world-championship-1950-2020`)
            - **Model Persistence**: Joblib, Keras `.keras` HDF5 format
            """
        )

    with t4:
        st.markdown(
            """
            ### Limitations & Future Enhancements
            - **High-frequency Telemetry (FastF1)**: Incorporating 10 Hz throttle, braking, and gear telemetry traces.
            - **Weather Telemetry**: Live track temperature, air humidity, and rain precipitation indices.
            - **Tire Compound Telemetry**: Direct encoding of Soft (C5/C4), Medium (C3), and Hard (C2/C1) degradation curves.
            - **Transformer / TCN Comparison**: Benchmarking Temporal Convolutional Networks (TCN) and Transformers against LSTM.
            """
        )
