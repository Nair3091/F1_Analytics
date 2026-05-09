"""
build_results_tables.py
-----------------------
Exports final production-grade CSV result tables from model and analytics outputs.

Creates:
    results/championship_predictions.csv
    results/race_predictions.csv
    results/feature_importance.csv
    results/constructor_dominance.csv
    results/driver_statistics.csv

Usage:
    python src/build_results_tables.py
"""

import logging
import os
import sys

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RACE_MODEL_DIR = os.path.join(RESULTS_DIR, "race_winner_random_forest")

RACE_FEATURE_COLS = [
    "grid",
    "quali_position",
    "rolling_avg_finish",
    "avg_finish_last_3",
    "constructor_avg_points",
    "avg_finish_at_circuit",
    "dnfs_last_5",
    "positions_gained",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def get_spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.builder \
        .appName("F1FinalResultsTables") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_csv(pdf, filename):
    output_path = os.path.join(RESULTS_DIR, filename)
    pdf.to_csv(output_path, index=False)
    log.info(f"Saved {output_path}")


def export_championship_predictions():
    path = os.path.join(RESULTS_DIR, "championship_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "championship_predictions.csv not found. Run src/champ_model.py first."
        )

    pdf = pd.read_csv(path).sort_values("year")
    write_csv(pdf, "championship_predictions.csv")


def export_race_predictions(spark):
    path = os.path.join(RESULTS_DIR, "race_winner_predictions.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "race_winner_predictions.parquet not found. Run src/race_winner_model.py first."
        )

    pdf = spark.read.parquet(path).orderBy(
        "year",
        "round",
        F.desc("winner_probability")
    ).toPandas()

    write_csv(pdf, "race_predictions.csv")


def export_feature_importance():
    if not os.path.exists(RACE_MODEL_DIR):
        raise FileNotFoundError(
            "Saved race winner model not found. Run src/race_winner_model.py first."
        )

    model = PipelineModel.load(RACE_MODEL_DIR)
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()

    pdf = pd.DataFrame({
        "feature": RACE_FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    total_importance = pdf["importance"].sum()
    if total_importance > 0:
        pdf["importance_pct"] = pdf["importance"] / total_importance
    else:
        pdf["importance_pct"] = 0.0

    write_csv(pdf, "feature_importance.csv")


def export_constructor_dominance(spark):
    path = os.path.join(RESULTS_DIR, "constructor_dominance.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "constructor_dominance.parquet not found. Run src/feature_engineering.py first."
        )

    pdf = spark.read.parquet(path).orderBy(F.desc("avg_points")).toPandas()
    write_csv(pdf, "constructor_dominance.csv")


def export_driver_statistics(spark):
    path = os.path.join(RESULTS_DIR, "driver_statistics.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "driver_statistics.parquet not found. Run src/feature_engineering.py first."
        )

    pdf = spark.read.parquet(path).orderBy(F.desc("wins")).toPandas()
    write_csv(pdf, "driver_statistics.csv")


def run():
    spark = get_spark()

    try:
        export_championship_predictions()
        export_race_predictions(spark)
        export_feature_importance()
        export_constructor_dominance(spark)
        export_driver_statistics(spark)
        log.info("=== Final results tables complete ===")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
