# 🏎️ FORMULA 1 LAP TIME PREDICTION USING DEEP LEARNING (LSTM)

**A Final-Year B.Tech Capstone Project in Computer Science & Engineering / AI & Data Science**

---

## 📌 Executive Summary

Formula 1 represents the pinnacle of motorsport engineering, where competitive margins and pit-stop strategies are determined by fractions of a second (milliseconds). In a typical 50-to-70-lap Grand Prix, lap-time variance is dictated by complex, non-linear dynamic variables: continuous tire compound degradation, progressive fuel mass burn-off (~1.7 kg/lap, equivalent to a ~0.08–0.12 second pace improvement per lap), changing aerodynamic track grip, and traffic interference.

This project designs, trains, evaluates, and deploys an end-to-end **Long Short-Term Memory (LSTM) Deep Learning System** capable of accurately forecasting a driver's next lap time based on their preceding 5-lap sequential stint. The system eliminates temporal data leakage through a strict chronological train/test split (training on 2019–2023 seasons and testing on the unseen 2024 season), achieving a **Mean Absolute Error (MAE) of ~2.28 seconds** across 23,861 unseen Grand Prix laps ($R^2 = 0.8075$).

The trained model is deployed within a Formula 1-themed **Streamlit** dashboard featuring live race telemetry replay, custom scenario simulation, exploratory data analytics, and performance visualizations.

---

## 🏗️ System Architecture

### Deep Learning Pipeline Overview

```
[ Historical F1 Kaggle Dataset (1950–2024) ]
                   │
                   ▼
[ Data Preprocessing & Merging Pipeline ]
  ├─ Filter outlier stints (55s <= Lap <= 160s)
  ├─ Contemporaneous Feature Selection (No Future Leakage)
  └─ Consecutive 5-Lap Windowing (t-4 to t)
                   │
                   ▼
[ Chronological Train/Val/Test Split ]
  ├─ Train Set: 2019–2023 (85,767 sequences)
  ├─ Val Set:   2023 Stints (14,391 sequences)
  └─ Test Set:  2024 Season (23,861 sequences)
                   │
                   ▼
[ Dual StandardScaler Transformation ]
  ├─ Feature Scaler: X (5 timesteps × 7 features)
  └─ Target Scaler:  y (Next Lap Milliseconds)
                   │
                   ▼
[ Stacked LSTM Neural Network ]
  ├─ Input Layer:  (None, 5, 7)
  ├─ LSTM Layer 1: 64 units, return_sequences=True (18,432 params)
  ├─ LSTM Layer 2: 32 units, return_sequences=False (12,416 params)
  ├─ Dense Layer:  32 units, ReLU activation (1,056 params)
  └─ Dense Output: 1 unit, Linear activation (33 params)
                   │
                   ▼
[ Inverse Transformation & Metrics Evaluation ]
  ├─ Formatted Lap Time (mm:ss.sss)
  ├─ MAE / RMSE / MSE / R²
  └─ Residuals & Error Distribution
                   │
                   ▼
[ Streamlit Interactive Web Application ]
  ├─ Page 1: Home Dashboard & Telemetry Health
  ├─ Page 2: Historical Data Analysis & EDA
  ├─ Page 3: Interactive Lap Time Predictor & Simulator
  ├─ Page 4: Model Performance & Error Analysis
  └─ Page 5: Academic Project Documentation
```

---

## 🧮 Mathematical Formulation of LSTM Cells

Standard Recurrent Neural Networks (RNNs) suffer from vanishing and exploding gradients when processing multi-lap sequences. The LSTM cell architecture resolves this through a cell state $C_t$ controlled by three non-linear gates:

1. **Forget Gate ($f_t$)**: Decides what proportion of the previous cell memory $C_{t-1}$ to discard:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate ($i_t$) & Candidate Memory ($\tilde{C}_t$)**: Determines what new lap telemetry features to encode into memory:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

3. **Cell State Update ($C_t$)**: Combines decayed prior memory with the newly gated candidate state:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

4. **Output Gate ($o_t$) & Hidden State ($h_t$)**: Extracts the filtered output representation passed to the next layer or dense regression head:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ represents the sigmoid activation function, and $\odot$ represents Hadamard element-wise multiplication.

---

## 📊 Dataset Specifications

The system utilizes the official Formula 1 World Championship dataset:
- **Source**: KaggleHub / Kaggle Dataset (`rohanrao/formula-1-world-championship-1950-2020` updated through 2024).
- **Core Files Utilized**:
  - `lap_times.csv`: 589,081 historical lap observations (`raceId`, `driverId`, `lap`, `position`, `time`, `milliseconds`).
  - `races.csv`: 1,125 Grand Prix events (`raceId`, `year`, `round`, `circuitId`, `name`, `date`).
  - `results.csv`: 26,759 driver race results (`grid`, `constructorId`).
  - `drivers.csv`: 861 official F1 drivers (`code`, `forename`, `surname`, `nationality`).
  - `circuits.csv`: 77 world championship circuits (`name`, `location`, `country`, `alt`).

### 🛡️ Anti-Data-Leakage Design
To guarantee scientific validity, the feature engineering pipeline strictly excludes any post-race or future information (such as final finishing position, race points, fastest lap awards, or status/retirement flags).

**Contemporaneous Features at Each Timestep $t$**:
1. `milliseconds`: Actual lap time of the current lap in the sequence.
2. `lap`: Current lap number in the race (proxy for fuel burn-off and track evolution).
3. `position`: Driver's live track position on lap $t$.
4. `circuitId`: Encoded circuit characteristics (track length, layout pace).
5. `driverId`: Driver identifier.
6. `constructorId`: Constructor/team vehicle performance baseline.
7. `grid`: Driver starting position on the grid.

---

## 📈 Empirical Evaluation Results

The model was evaluated against **23,861 unseen test sequence samples from the 2024 Formula 1 World Championship season**.

| Metric | Measured Value (Milliseconds) | Measured Value (Seconds) | Description |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **2,281.44 ms** | **2.281 s** | Average prediction deviation per lap |
| **Root Mean Squared Error (RMSE)** | **5,535.87 ms** | **5.536 s** | Penalizes severe prediction outliers |
| **Mean Squared Error (MSE)** | **30,645,818.0** | **30.65 s²** | Loss convergence criterion |
| **Coefficient of Determination ($R^2$)** | **0.8075** | **80.75%** | Percentage of lap time variance explained |
| **Mean Absolute Percentage Error (MAPE)** | **2.52%** | **2.52%** | Average relative percentage error |
| **Test Dataset Size** | **23,861 laps** | **24 Grands Prix** | Unseen 2024 modern F1 season |

### Real Telemetry Validation Sample (2024 Bahrain Grand Prix)
- **Driver**: Fernando Alonso (Aston Martin)
- **Target Lap**: Lap 54
- **Preceding 5-Lap Sequence**: `1:34.571`, `1:34.677`, `1:34.916`, `1:34.862`, `1:35.249`
- **Model Predicted Lap Time**: `1:36.011` (96,011 ms)
- **Actual Ground Truth Lap Time**: `1:35.745` (95,745 ms)
- **Error Delta**: **-0.266 seconds (-266 milliseconds)**
- **Prediction Accuracy**: **99.72%**

---

## 🗂️ Project Directory Structure

```
F1_laptime_prediction/
│
├── app.py                      # Main Streamlit 5-page dashboard application
├── train_model.py              # LSTM model builder, compiler, and training pipeline
├── data_processing.py          # KaggleHub download, merging, windowing, and scalers
├── prediction.py               # Real-time inference engine and scenario simulator
├── evaluation.py               # Evaluation metrics, residual calculations, CSV exports
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation
│
├── models/                     # Saved deep learning artifacts
│   ├── lstm_model.keras        # Trained TensorFlow/Keras LSTM model
│   ├── scaler_X.pkl            # StandardScaler for 7 input features
│   ├── scaler_y.pkl            # StandardScaler for target milliseconds
│   ├── scaler.pkl              # Unified scaler bundle
│   ├── feature_columns.json    # JSON list of feature names
│   └── metadata.json           # Model configuration and training metadata
│
├── data/                       # Kaggle F1 CSV datasets
│   ├── lap_times.csv
│   ├── races.csv
│   ├── results.csv
│   ├── drivers.csv
│   └── circuits.csv
│
├── outputs/                    # Training and evaluation logs
│   ├── training_history.json   # Epoch-by-epoch loss & MAE curves
│   ├── evaluation_results.json # Verified MAE, RMSE, MSE, and R² metrics
│   ├── test_predictions.csv    # 23,861 test samples with actual vs predicted times
│   ├── X_test.npy              # Test input feature tensor
│   ├── y_test.npy              # Test ground-truth target vector
│   └── meta_test.csv           # Test sample race/driver metadata
│
└── assets/                     # UI styling assets
    └── style.css               # Formula 1 dark & red styling stylesheet
```

---

## 💻 Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Pip package manager

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd c:\Users\Venkat\OneDrive\Desktop\F1_laptime_prediction
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Model Training (Optional - Pretrained Model Provided)
To retrain the LSTM network from scratch:
```powershell
python train_model.py --epochs 15 --batch-size 128 --seq-len 5
```

### Step 4: Run Model Evaluation
To re-evaluate the model against the 2024 test season and generate performance tables:
```powershell
python evaluation.py
```

### Step 5: Launch the Streamlit Web Application
**Option A: One-Click Windows Launch (Recommended)**
Simply double-click the created **`run_app.bat`** file in the project folder.

**Option B: Via PowerShell / Terminal**
```powershell
python -m streamlit run app.py --server.port 8501
```
Open your web browser and navigate to `http://localhost:8501`.

---

## 🖥️ Web Application Pages Overview

1. **🏠 Page 1: Home Dashboard**: High-level telemetry statistics, dataset file verification status, active model status, and key performance highlights.
2. **📊 Page 2: Data Analysis & EDA**: Interactive race pace progression charts, driver boxplots, circuit pace comparisons across 77 historical tracks.
3. **⏱️ Page 3: Lap Time Prediction**: 
   - *Mode A (Historical Telemetry)*: Pick any race, driver, and lap number to run real inference against ground truth.
   - *Mode B (Custom Simulator)*: Adjust circuit pace, current lap, position, and 5 custom lap times to forecast vehicle behavior.
4. **📈 Page 4: Model Performance**: Training vs validation loss learning curves, actual vs predicted scatter plots with identity line ($y=x$), residuals distribution, and sample prediction inspector.
5. **ℹ️ Page 5: About Project**: Academic background, mathematical foundations of LSTM gating, methodology, limitations, and future work.

---

## 👥 Academic Attribution & Credits
- **Course**: Final Year B.Tech Project
- **Specialization**: Artificial Intelligence & Data Science / Computer Science Engineering
- **Dataset**: Rohan Rao (Formula 1 World Championship 1950–2024, Kaggle)
- **Deep Learning Framework**: TensorFlow & Keras
- **Visualization & UI**: Streamlit & Plotly
