import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
DATA_FEATURES   = os.path.join(BASE_DIR, "data", "features")

OPENF1_RAW      = os.path.join(DATA_RAW, "openf1")
ERGAST_RAW      = os.path.join(DATA_RAW, "ergast")

# ── OpenF1 API ─────────────────────────────────────────────────────────────
OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_YEARS    = [2023, 2024, 2025]

# Endpoints to fetch per race session
OPENF1_ENDPOINTS = [
    "laps",
    "stints",
    "pit",
    "intervals",
    "weather",
    "race_control",
    "position",
    "drivers",
]

# ── Ergast ─────────────────────────────────────────────────────────────────
ERGAST_BASE_URL     = "https://ergast.com/api/f1"
ERGAST_DOWNLOAD_URL = None
ERGAST_YEARS        = (1950, 2022)   # inclusive

# CSV files we care about from the Ergast dump
ERGAST_FILES = [
    "races.csv",
    "results.csv",
    "drivers.csv",
    "constructors.csv",
    "qualifying.csv",
    "pit_stops.csv",
    "lap_times.csv",
    "driver_standings.csv",
    "constructor_standings.csv",
    "circuits.csv",
    "status.csv",
]

# ── Spark ──────────────────────────────────────────────────────────────────
SPARK_APP_NAME          = "F1Predictor"
SPARK_SHUFFLE_PARTITIONS = 50

# ── Model ──────────────────────────────────────────────────────────────────
TRAIN_YEARS_CUTOFF  = 2018   # train on data up to this year
TEST_YEARS          = [2019,2020,2021,2022,2023, 2024, 2025]

RACE_TARGET_COL  = "is_winner"      # binary: did this driver win the race?
CHAMP_TARGET_COL = "won_championship"  # binary: did this driver win the title?

RACE_FEATURE_COLS = [
    "grid_position",
    "gap_to_pole",
    "rolling_avg_finish",
    "rolling_points",
    "rolling_wins",
    "constructor_rolling_points",
    "avg_pit_stop_ms",
    "num_stints",
    "starting_compound_enc",
    "rain_flag",
    "avg_track_temp",
    "safety_car_count",
    "circuit_enc",
]

CHAMP_FEATURE_COLS = [
    "race_position",
    "points_after_race",
    "gap_to_leader_standings",
    "rolling_wins",
    "rolling_dnfs",
    "constructor_points",
    "races_remaining",
    "round_number",
]

# ── Misc ───────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
LOG_LEVEL       = "INFO"