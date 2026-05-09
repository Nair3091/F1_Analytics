# F1 Analytics Project — Complete Handoff Document

---

## Project Overview

A Big Data Analytics (BDA) project that predicts:
1. **Race winner per Grand Prix** (per circuit, pre-race prediction)
2. **Drivers' Championship winner** (mid-season / end-of-season prediction)

Built with Apache Spark for data processing, XGBoost for race winner prediction, and a PyTorch Transformer for championship prediction.

---

## Environment

- **OS:** Windows
- **Python:** 3.12.3 (virtual environment: `f1_env`)
- **PySpark:** 4.1.1
- **Java (JDK):** 21 (OpenJDK)
- **Hadoop:** 3.3.5 (winutils installed at `C:\Hadoop\bin\`)
- **HADOOP_HOME:** `C:\Hadoop` (set as system environment variable)

### Virtual Environment
Located at: `C:\College\Amrita\Sem-6\BDA\F1_Project\f1_env\`

Activate with:
```bash
f1_env\Scripts\activate
```

### Installed Packages
```
pyspark==4.1.1
torch, torchvision
xgboost
lightgbm
aiohttp
requests
pandas, numpy, scikit-learn
matplotlib, seaborn, plotly
jupyter, notebook, ipykernel
mlflow
```

---

## Project Structure

```
F1_Project/
├── data/
│   ├── raw/
│   │   ├── ergast/          # Kaggle F1 CSVs (manually downloaded)
│   │   └── openf1/          # OpenF1 JSON files (fetched by script)
│   │       ├── 2023/
│   │       │   ├── {session_key}/
│   │       │   │   ├── session_meta.json
│   │       │   │   ├── laps.json
│   │       │   │   ├── stints.json
│   │       │   │   ├── pit.json
│   │       │   │   ├── weather.json
│   │       │   │   ├── race_control.json
│   │       │   │   ├── position.json
│   │       │   │   └── drivers.json
│   │       │   └── quali_{session_key}/
│   │       │       └── laps.json
│   │       ├── 2024/        # (fetched but may be incomplete)
│   │       └── 2025/        # (fetched but may be incomplete)
│   ├── processed/
│   │   └── openf1_features.parquet   # OpenF1 intermediate features
│   └── features/
│       ├── race_features.parquet     # 26,759 rows — main ML input
│       └── champ_features.parquet    # 26,759 rows — championship ML input
├── src/
│   ├── fetch_data.py         # DONE — data ingestion
│   ├── features.py           # DONE — Spark feature engineering
│   ├── race_model.py         # DONE — XGBoost race winner model
│   └── champ_model.py        # DONE — PyTorch Transformer championship model
├── results/
│   ├── race_winner_predictions.csv
│   ├── feature_importance_race.png
│   ├── confusion_matrix_race.png
│   ├── accuracy_by_year_race.png
│   ├── championship_predictions.csv   # generated after champ_model.py runs
│   ├── champ_loss_curve.png           # generated after champ_model.py runs
│   ├── champ_predictions_by_year.png  # generated after champ_model.py runs
│   └── champ_transformer.pt           # saved model weights
├── notebooks/               # EMPTY — not yet built
├── config.py                # DONE — all project constants
└── requirements.txt         # NOT YET WRITTEN
```

---

## Data Sources

### 1. Ergast (1950–2022) — Historical Data
- **Status:** DONE
- **Source:** Kaggle — https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
- **Note:** Ergast API is permanently shut down. Data must be manually downloaded from Kaggle and placed in `data/raw/ergast/`
- **Files used:**
  - races.csv, results.csv, drivers.csv, constructors.csv
  - qualifying.csv, pit_stops.csv, lap_times.csv
  - driver_standings.csv, constructor_standings.csv
  - circuits.csv, status.csv

### 2. OpenF1 API (2023–2025) — Granular Session Data
- **Status:** 2023 fully fetched; 2024 and 2025 may be incomplete due to rate limiting
- **Base URL:** https://api.openf1.org/v1
- **Endpoints fetched per race session:** laps, stints, pit, intervals, weather, race_control, position, drivers
- **Qualifying laps** fetched separately for grid position features
- **Known issue:** Some sessions returned 0 records due to rate limiting during ingestion. If 2024/2025 data is sparse, re-run `fetch_data.py` with `OPENF1_YEARS = [2024, 2025]` in config.py

---

## config.py — Key Constants

```python
# Paths
BASE_DIR, DATA_RAW, DATA_PROCESSED, DATA_FEATURES
OPENF1_RAW = data/raw/openf1/
ERGAST_RAW = data/raw/ergast/

# OpenF1
OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_YEARS = [2023, 2024, 2025]
OPENF1_ENDPOINTS = ["laps", "stints", "pit", "intervals", "weather",
                    "race_control", "position", "drivers"]

# Ergast
ERGAST_FILES = [list of CSV filenames]

# Spark
SPARK_APP_NAME = "F1Predictor"
SPARK_SHUFFLE_PARTITIONS = 50

# Model
TRAIN_YEARS_CUTOFF = 2022      # train on <=2022, test on >2022
TEST_YEARS = [2023, 2024, 2025]
RACE_TARGET_COL = "is_winner"
CHAMP_TARGET_COL = "won_championship"

RACE_FEATURE_COLS = [list of feature column names]
CHAMP_FEATURE_COLS = [list of feature column names]

RANDOM_SEED = 42
```

---

## src/fetch_data.py — Data Ingestion

**Status:** DONE and working

**What it does:**
- Step 1: Verifies all Ergast CSV files exist in `data/raw/ergast/` (no download — manual)
- Step 2: Fetches all OpenF1 race session data for 2023–2025 asynchronously
  - Saves each endpoint as a JSON file under `data/raw/openf1/{year}/{session_key}/{endpoint}.json`
  - Saves session metadata as `session_meta.json`
  - Skip logic: if all endpoint files already exist for a session, skips it
- Step 3: Fetches qualifying lap data separately, saved under `data/raw/openf1/{year}/quali_{session_key}/laps.json`

**Key implementation details:**
- Uses `aiohttp` with async/await for fetching
- Sequential endpoint fetching per session (NOT parallel) to avoid rate limits
- Exponential backoff: 3s, 6s, 12s, 24s, 48s on 429 responses
- 2s delay between endpoints, 5s between sessions, 30s between years
- 5 retries for session list fetches, 3 retries for endpoint fetches
- Empty files (size <= 10 bytes) are skipped on reload

**Run:**
```bash
python src\fetch_data.py
```

---

## src/features.py — Spark Feature Engineering

**Status:** DONE and working

**What it does:**
Loads raw data → builds features → saves two parquet files.

**Spark configuration used (critical for Windows):**
```python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"] = "C:\\Hadoop"
SparkSession.builder.master("local[2]")
  .config("spark.sql.shuffle.partitions", "8")
  .config("spark.driver.memory", "4g")
  .config("spark.driver.maxResultSize", "2g")
  .config("spark.default.parallelism", "8")
```

**Features built from Ergast:**
- `grid_position` — from qualifying.csv, falls back to grid column
- `gap_to_pole` — grid_position minus best in race
- `final_position`, `points`, `dnf` flag
- `rolling_avg_finish` — avg finish position over last 5 races (window function)
- `rolling_points` — sum of points over last 5 races
- `rolling_wins` — wins in last 5 races
- `rolling_dnfs` — DNFs in last 5 races
- `constructor_rolling_points` — team's last 5 races points
- `driver_circuit_starts` — how many times driver has raced at this circuit historically
- `driver_circuit_wins` — how many times driver has won at this circuit
- `num_pit_stops`, `avg_pit_ms`, `first_pit_lap`
- `is_winner` (target), `is_podium`

**Features built from OpenF1:**
- `rain_flag` — 1 if rainfall > 0 during race
- `avg_air_temp`, `avg_track_temp`, `avg_wind_speed`
- `safety_car_count` — number of SC/VSC deployments
- `num_stints`, `avg_stint_length` (lap_end - lap_start), `starting_compound`
- `starting_compound_enc` — SOFT=1, MEDIUM=2, HARD=3, INTERMEDIATE=4, WET=5
- `num_pit_stops_f1`, `avg_pit_duration_s`
- `grid_position_f1`, `gap_to_pole_f1` — from actual qualifying lap times

**Important OpenF1 column names (actual API response):**
- Sessions: `session_key, meeting_key, year, session_name, session_type, location, country_name`
  - NOTE: field is `session_name` NOT `meeting_name`
- Stints: `compound, driver_number, lap_end, lap_start, meeting_key, session_key, stint_number, tyre_age_at_start`
  - NOTE: field is `lap_start`/`lap_end` NOT `lap_count`
- Pit: `pit_duration` (in seconds)
- Weather: `air_temperature, track_temperature, rainfall, wind_speed`
- Race control: `category` (values: "SafetyCar", "VirtualSafetyCar", "Flag", etc.)

**Output:**
- `data/features/race_features.parquet` — 26,759 rows (one per driver per race, 1950–2024)
- `data/features/champ_features.parquet` — 26,759 rows (same rows + cumulative season stats)
- `data/processed/openf1_features.parquet` — OpenF1 intermediate

**Note:** Currently race_features only contains Ergast data with null OpenF1 columns. The OpenF1 features are saved separately. A future improvement is to join OpenF1 features onto Ergast rows for 2023+ races.

**Run:**
```bash
python src\features.py
```

---

## src/race_model.py — Race Winner Prediction

**Status:** DONE and working

**Model:** XGBoost binary classifier
**Target:** `is_winner` — did this driver win the race?
**Train/test split:** Temporal — train ≤2022, test >2022

**17 features used:**
```
grid_position, gap_to_pole, rolling_avg_finish, rolling_points,
rolling_wins, rolling_dnfs, constructor_rolling_points,
driver_circuit_starts, driver_circuit_wins, num_pit_stops,
avg_pit_ms, rain_flag, avg_track_temp, safety_car_count,
num_stints, starting_compound_enc, circuit_enc
```

**XGBoost hyperparameters:**
```python
n_estimators=300, max_depth=5, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
scale_pos_weight=22.9  # handles class imbalance (1 winner per ~20 drivers)
```

**Results:**
- ROC-AUC: **0.9621**
- Race-level winner prediction accuracy: **60.87%**
- Train rows: 25,840 | Test rows: 919

**Evaluation approach:**
- Driver-level: standard ROC-AUC, precision/recall, confusion matrix
- Race-level: for each race, pick driver with highest win probability → check if correct

**Output files:**
- `results/race_winner_predictions.csv` — columns: year, round, predicted, actual, win_prob, correct
- `results/feature_importance_race.png`
- `results/confusion_matrix_race.png`
- `results/accuracy_by_year_race.png`

**Run:**
```bash
python src\race_model.py
```

---

## src/champ_model.py — Championship Winner Prediction

**Status:** WRITTEN, not yet confirmed running successfully

**Model:** PyTorch Transformer Encoder
**Target:** `won_championship` — did this driver win the title that year?
**Train/test split:** Temporal — train ≤2022, val = 2018–2022, test >2022

**Architecture:**
```
Input: (batch, MAX_ROUNDS=25, feat_dim=8)  — sequence of race results per season
  → Linear projection to d_model
  → TransformerEncoder (2 layers, 4 heads, dropout=0.2)
  → Mean pool across rounds
  → Linear(d_model, 32) → ReLU → Dropout → Linear(32, 1) → Sigmoid
Output: (batch,) — championship win probability
```

**8 sequence features per round:**
```
final_position, points, cumulative_points, cumulative_wins,
gap_to_leader, races_remaining, rolling_avg_finish, rolling_dnfs
```

**Training config:**
```python
EPOCHS=40, LR=1e-3, BATCH_SIZE=32
Optimizer: Adam with weight_decay=1e-4
Scheduler: StepLR(step_size=15, gamma=0.5)
Loss: BCELoss
Gradient clipping: max_norm=1.0
```

**Sequence building:**
- One sequence per (driver, year)
- Padded/truncated to MAX_ROUNDS=25
- Normalized using train set mean/std

**Evaluation:**
- Driver/season level: ROC-AUC
- Season level: pick driver with highest probability per year → check if correct

**Output files (after running):**
- `results/championship_predictions.csv` — year, predicted, actual, prob, correct
- `results/champ_loss_curve.png`
- `results/champ_predictions_by_year.png`
- `results/champ_transformer.pt` — saved model weights

**Run:**
```bash
python src\champ_model.py
```

---

## What Is Done

| Task | Status |
|---|---|
| Project structure setup | DONE |
| Virtual environment setup | DONE |
| Ergast CSV data download | DONE (manual from Kaggle) |
| OpenF1 2023 data ingestion | DONE (fully fetched) |
| OpenF1 2024/2025 ingestion | PARTIAL (rate limited, may need re-run) |
| Spark feature engineering | DONE |
| Race features parquet | DONE (26,759 rows) |
| Championship features parquet | DONE (26,759 rows) |
| Race winner model (XGBoost) | DONE (AUC=0.96, accuracy=60.87%) |
| Championship model (Transformer) | WRITTEN — needs test run |
| Jupyter notebooks | NOT STARTED |
| requirements.txt | NOT WRITTEN |
| README.md | NOT WRITTEN |

---

## What Is Left

### 1. Run and validate champ_model.py
```bash
python src\champ_model.py
```
Check output metrics. If accuracy is low, possible improvements:
- Increase EPOCHS to 60-80
- Add more features (constructor_rolling_points, is_home_race)
- Try XGBoost instead of Transformer for comparison

### 2. Fix OpenF1 join in features.py (optional but improves 2023+ predictions)
Currently, race_features.parquet has null values for OpenF1 columns (rain_flag, track_temp, etc.)
for all rows because the join between Ergast and OpenF1 was not implemented.
To fix: match OpenF1 sessions to Ergast races via year + round number or circuit name,
then enrich the 2023+ rows with OpenF1 weather/safety car/stint data.

### 3. Re-fetch 2024/2025 OpenF1 data if incomplete
In config.py set `OPENF1_YEARS = [2024, 2025]` then run:
```bash
python src\fetch_data.py
```
The skip logic will only fetch missing sessions.

### 4. Write requirements.txt
```bash
pip freeze > requirements.txt
```

### 5. Build Jupyter notebooks (one per pipeline stage)
- `notebooks/01_data_ingestion.ipynb` — explore raw data, check counts
- `notebooks/02_feature_engineering.ipynb` — EDA on feature distributions
- `notebooks/03_race_winner_model.ipynb` — model training + visualizations
- `notebooks/04_championship_model.ipynb` — transformer training + results

### 6. Write README.md
Should cover: setup steps, how to run each script, results summary

### 7. Optional improvements
- **Per-circuit models:** train separate XGBoost model per circuit (Monaco, Monza, etc.)
- **Hyperparameter tuning:** GridSearchCV or Optuna for XGBoost
- **SHAP values:** explainability for race model predictions
- **Streamlit/Dash dashboard:** interactive visualization of predictions
- **Mid-season championship prediction:** run champ_model after each round to update predictions

---

## Common Issues and Fixes

### "Python worker failed to connect back"
**Cause:** Spark can't find Python executable
**Fix:** Add at top of script before SparkSession:
```python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
```

### "ModuleNotFoundError: No module named config"
**Cause:** src/ scripts can't find config.py in parent directory
**Fix:** Add at very top of every src/ script:
```python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Spark crashes with "Python worker exited unexpectedly"
**Cause:** Too many parallel tasks + insufficient memory
**Fix:** Use `master("local[2]")` and reduce parallelism:
```python
.config("spark.sql.shuffle.partitions", "8")
.config("spark.default.parallelism", "8")
```

### "UNRESOLVED_COLUMN" errors in features.py
**Cause:** OpenF1 API column names differ from what was assumed
**Known mappings:**
- `meeting_name` → `session_name`
- `lap_count` → use `lap_end - lap_start`

### Ergast download 520 error
**Cause:** Ergast API is permanently shut down
**Fix:** Download from Kaggle manually instead

### OpenF1 429 rate limiting
**Cause:** Too many concurrent requests
**Fix:** Sequential fetching with delays (already implemented in current fetch_data.py)

---

## How to Run the Full Pipeline (Fresh Start)

```bash
# 1. Activate environment
cd C:\College\Amrita\Sem-6\BDA\F1_Project
f1_env\Scripts\activate

# 2. Fetch OpenF1 data (Ergast must already be in data/raw/ergast/)
python src\fetch_data.py

# 3. Build features
python src\features.py

# 4. Train race winner model
python src\race_model.py

# 5. Train championship model
python src\champ_model.py
```

---

## Results Summary

| Model | Metric | Value |
|---|---|---|
| Race Winner (XGBoost) | ROC-AUC | 0.9621 |
| Race Winner (XGBoost) | Per-race accuracy | 60.87% |
| Championship (Transformer) | ROC-AUC | TBD (not yet run) |
| Championship (Transformer) | Per-season accuracy | TBD (not yet run) |
