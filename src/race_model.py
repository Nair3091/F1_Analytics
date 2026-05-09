"""
race_model.py
─────────────
Predicts the race winner for each Grand Prix using XGBoost.
Trained on Ergast historical data (1950–2022), evaluated on recent seasons.

Usage:
    python src/race_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import (
    DATA_FEATURES, TRAIN_YEARS_CUTOFF, TEST_YEARS,
    RACE_FEATURE_COLS, RACE_TARGET_COL,
    RANDOM_SEED, LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Spark Session ──────────────────────────────────────────────────────────

def get_spark():
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["HADOOP_HOME"]           = "C:\\Hadoop"

    spark = SparkSession.builder \
        .appName("F1RaceModel") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ── Feature Preparation ────────────────────────────────────────────────────

# Actual feature columns available in race_features.parquet
# (subset of RACE_FEATURE_COLS that exist after Ergast-only build)
AVAILABLE_FEATURES = [
    "grid_position",
    "gap_to_pole",
    "rolling_avg_finish",
    "rolling_points",
    "rolling_wins",
    "rolling_dnfs",
    "constructor_rolling_points",
    "driver_circuit_starts",
    "driver_circuit_wins",
    "num_pit_stops",
    "avg_pit_ms",
    "rain_flag",
    "avg_track_temp",
    "safety_car_count",
    "num_stints",
    "starting_compound_enc",
]


def load_and_prepare(spark):
    log.info("Loading race features...")

    path = os.path.join(DATA_FEATURES, "race_features.parquet")
    df = spark.read.parquet(path)

    # Encode circuit_id as integer
    df = df.withColumn("circuit_enc",
        F.dense_rank().over(
            __import__("pyspark.sql.window", fromlist=["Window"])
            .Window.orderBy("circuit_id")
        )
    )

    # Convert to pandas for sklearn/xgboost
    cols_to_pull = AVAILABLE_FEATURES + ["circuit_enc", "circuit_id",
                                          "driver_name", "year", "round",
                                          "race_id", RACE_TARGET_COL]

    # Only keep columns that actually exist
    existing = df.columns
    cols_to_pull = [c for c in cols_to_pull if c in existing]

    pdf = df.select(cols_to_pull).toPandas()

    # Add circuit encoding if not already present
    if "circuit_enc" not in pdf.columns:
        le = LabelEncoder()
        pdf["circuit_enc"] = le.fit_transform(pdf["circuit_id"].fillna("unknown"))

    # Fill any remaining nulls
    for col in AVAILABLE_FEATURES + ["circuit_enc"]:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0)

    log.info(f"Loaded {len(pdf)} rows, {pdf[RACE_TARGET_COL].sum()} winners")
    return pdf


def get_feature_cols(pdf):
    """Return only the feature columns that exist in the dataframe."""
    candidates = AVAILABLE_FEATURES + ["circuit_enc"]
    return [c for c in candidates if c in pdf.columns]


# ── Temporal Train/Test Split ──────────────────────────────────────────────

def split_data(pdf):
    train = pdf[pdf["year"] <= TRAIN_YEARS_CUTOFF].copy()
    test  = pdf[pdf["year"] >  TRAIN_YEARS_CUTOFF].copy()
    log.info(f"Train: {len(train)} rows ({train['year'].min()}–{train['year'].max()})")
    log.info(f"Test:  {len(test)} rows  ({test['year'].min()}–{test['year'].max()})")
    return train, test


# ── Model Training ─────────────────────────────────────────────────────────

def train_model(train, feature_cols):
    log.info("Training XGBoost race winner model...")

    X = train[feature_cols]
    y = train[RACE_TARGET_COL]

    # Class imbalance — ~1 winner per 20 drivers
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    log.info(f"Class imbalance ratio: {scale_pos_weight:.1f} — applied to model")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X, y)
    log.info("Model training complete.")
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, test, feature_cols):
    log.info("Evaluating model...")

    X_test = test[feature_cols]
    y_test = test[RACE_TARGET_COL]

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    log.info(f"ROC-AUC: {auc:.4f}")
    log.info("\n" + classification_report(y_test, preds, target_names=["Not Winner", "Winner"]))

    return probs, preds, auc


def predict_race_winners(model, test, feature_cols):
    """
    For each race in the test set, pick the driver with the highest
    win probability as the predicted winner.
    """
    test = test.copy()
    test["win_prob"] = model.predict_proba(test[feature_cols])[:, 1]

    results = []
    for (year, round_num), group in test.groupby(["year", "round"]):
        predicted = group.loc[group["win_prob"].idxmax()]
        actual    = group[group[RACE_TARGET_COL] == 1]

        predicted_name = predicted.get("driver_name", predicted.get("driver_id", "?"))
        actual_name    = actual["driver_name"].values[0] if (
            "driver_name" in actual.columns and len(actual) > 0
        ) else "?"

        correct = int(predicted_name == actual_name)
        results.append({
            "year":           year,
            "round":          round_num,
            "predicted":      predicted_name,
            "actual":         actual_name,
            "win_prob":       round(predicted["win_prob"], 3),
            "correct":        correct,
        })

    results_df = pd.DataFrame(results).sort_values(["year", "round"])
    accuracy = results_df["correct"].mean()
    log.info(f"Race-level winner prediction accuracy: {accuracy:.2%}")

    # Save predictions
    out_path = os.path.join(RESULTS_DIR, "race_winner_predictions.csv")
    results_df.to_csv(out_path, index=False)
    log.info(f"Predictions saved to {out_path}")

    return results_df, accuracy


# ── Feature Importance Plot ────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance — Race Winner Model", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "feature_importance_race.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Feature importance plot saved to {out_path}")


def plot_confusion_matrix(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Winner", "Winner"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Race Winner Model")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confusion_matrix_race.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved to {out_path}")


def plot_accuracy_by_year(results_df):
    yearly = results_df.groupby("year")["correct"].mean().reset_index()
    yearly.columns = ["year", "accuracy"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly["year"], yearly["accuracy"], color="steelblue")
    ax.axhline(results_df["correct"].mean(), color="red",
               linestyle="--", label=f"Overall: {results_df['correct'].mean():.2%}")
    ax.set_title("Race Winner Prediction Accuracy by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "accuracy_by_year_race.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Accuracy by year plot saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    spark = get_spark()

    # Load data
    pdf = load_and_prepare(spark)
    spark.stop()

    # Split
    train, test = split_data(pdf)
    feature_cols = get_feature_cols(pdf)
    log.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Train
    model = train_model(train, feature_cols)

    # Evaluate
    probs, preds, auc = evaluate_model(model, test, feature_cols)

    # Per-race winner predictions
    results_df, accuracy = predict_race_winners(model, test, feature_cols)

    # Plots
    plot_feature_importance(model, feature_cols)
    plot_confusion_matrix(test[RACE_TARGET_COL], preds)
    plot_accuracy_by_year(results_df)

    log.info("=== Race model complete ===")
    log.info(f"ROC-AUC:          {auc:.4f}")
    log.info(f"Winner accuracy:  {accuracy:.2%}")
    log.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    run()