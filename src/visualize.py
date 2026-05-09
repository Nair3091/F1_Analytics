"""
visualize.py
------------
Visual analytics for the F1 big-data feature pipeline.

Generates:
1. Driver momentum graph
2. Constructor dominance chart
3. Championship probability curves
4. Circuit DNF analysis

Usage:
    python src/visualize.py
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


DRIVER_MOMENTUM_NAMES = ["Verstappen", "Hamilton", "Alonso"]
CONSTRUCTOR_NAMES = ["Ferrari", "Mercedes", "Red Bull", "McLaren"]


def get_spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.builder \
        .appName("F1VisualAnalytics") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def save_plot(fig, filename):
    output_path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    log.info(f"Saved {output_path}")


def load_features(spark):
    champ_path = os.path.join(FEATURE_DIR, "champ_features.parquet")
    race_path = os.path.join(FEATURE_DIR, "race_features.parquet")
    circuit_stats_path = os.path.join(RESULTS_DIR, "circuit_statistics.parquet")

    log.info(f"Loading championship features from {champ_path}")
    champ_df = spark.read.parquet(champ_path)

    log.info(f"Loading race features from {race_path}")
    race_df = spark.read.parquet(race_path)

    log.info(f"Loading circuit statistics from {circuit_stats_path}")
    circuit_stats_df = spark.read.parquet(circuit_stats_path)

    return champ_df, race_df, circuit_stats_df


def plot_driver_momentum(champ_df):
    log.info("Generating driver momentum graph...")

    latest_year = champ_df.agg(F.max("year").alias("latest_year")).first()["latest_year"]
    name_filter = None
    for name in DRIVER_MOMENTUM_NAMES:
        condition = F.col("driver_name").contains(name)
        name_filter = condition if name_filter is None else name_filter | condition

    pdf = champ_df.filter(
        (F.col("year") == latest_year) & name_filter
    ).select(
        "year",
        "round",
        "driver_name",
        "rolling_avg_finish"
    ).orderBy("driver_name", "round").toPandas()

    fig, ax = plt.subplots(figsize=(10, 5))
    for driver_name, group in pdf.groupby("driver_name"):
        ax.plot(
            group["round"],
            group["rolling_avg_finish"],
            marker="o",
            linewidth=2,
            label=driver_name
        )

    ax.invert_yaxis()
    ax.set_title(f"Driver Momentum: Rolling Average Finish ({latest_year})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Rolling Average Finish")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_plot(fig, "driver_momentum.png")


def plot_constructor_dominance(champ_df):
    log.info("Generating constructor dominance chart...")

    constructor_df = champ_df.filter(
        F.col("constructor_name").isin(CONSTRUCTOR_NAMES)
    ).groupBy("constructor_name").agg(
        F.avg("points").alias("avg_points"),
        F.sum(F.when(F.col("final_position") == 1, 1).otherwise(0)).alias("total_wins")
    ).orderBy(F.desc("avg_points"))

    pdf = constructor_df.toPandas()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = range(len(pdf))
    ax1.bar(
        [i - 0.18 for i in x],
        pdf["avg_points"],
        width=0.36,
        label="Average Points",
        color="#2f6f9f"
    )
    ax2.bar(
        [i + 0.18 for i in x],
        pdf["total_wins"],
        width=0.36,
        label="Total Wins",
        color="#c94c4c"
    )

    ax1.set_title("Constructor Dominance")
    ax1.set_xlabel("Constructor")
    ax1.set_ylabel("Average Points")
    ax2.set_ylabel("Total Wins")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(pdf["constructor_name"], rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    save_plot(fig, "constructor_dominance.png")


def plot_championship_probability_curves(champ_df):
    log.info("Generating championship probability curves...")

    latest_year = champ_df.agg(F.max("year").alias("latest_year")).first()["latest_year"]

    round_window = Window.partitionBy("year", "round")
    ranked_df = champ_df.filter(F.col("year") == latest_year).withColumn(
        "season_round_points",
        F.max("cumulative_points").over(round_window)
    ).withColumn(
        "point_share",
        F.col("cumulative_points") / F.when(
            F.col("season_round_points") == 0,
            F.lit(1)
        ).otherwise(F.col("season_round_points"))
    ).withColumn(
        "championship_probability",
        F.pow(F.col("point_share"), F.lit(2.0))
    )

    top_driver_window = Window.partitionBy("driver_name")
    top_drivers = ranked_df.withColumn(
        "max_probability",
        F.max("championship_probability").over(top_driver_window)
    ).select("driver_name", "max_probability").distinct() \
        .orderBy(F.desc("max_probability")).limit(6)

    pdf = ranked_df.join(top_drivers, on="driver_name", how="inner").select(
        "round",
        "driver_name",
        "championship_probability"
    ).orderBy("driver_name", "round").toPandas()

    fig, ax = plt.subplots(figsize=(10, 5))
    for driver_name, group in pdf.groupby("driver_name"):
        ax.plot(
            group["round"],
            group["championship_probability"],
            marker="o",
            linewidth=2,
            label=driver_name
        )

    ax.set_title(f"Championship Probability Curves ({latest_year})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Estimated Championship Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_plot(fig, "championship_probability_curves.png")


def plot_dnf_analysis(circuit_stats_df):
    log.info("Generating DNF analysis...")

    circuit_df = circuit_stats_df.select(
        "circuit_name",
        F.col("avg_dnf_rate").alias("dnf_rate")
    ).orderBy(F.desc("dnf_rate")).limit(12)

    pdf = circuit_df.toPandas().sort_values("dnf_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pdf["circuit_name"], pdf["dnf_rate"], color="#6a8f3a")
    ax.set_title("Circuits With Highest DNF Rates")
    ax.set_xlabel("DNF Rate")
    ax.set_ylabel("Circuit")
    ax.grid(True, axis="x", alpha=0.25)

    save_plot(fig, "dnf_analysis.png")


def run():
    spark = get_spark()

    try:
        champ_df, race_df, circuit_stats_df = load_features(spark)
        plot_driver_momentum(champ_df)
        plot_constructor_dominance(champ_df)
        plot_championship_probability_curves(champ_df)
        plot_dnf_analysis(circuit_stats_df)
        log.info("=== Visual analytics complete ===")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
