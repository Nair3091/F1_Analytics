# feature_engineering.py
"""
feature_engineering.py
──────────────────────
Distributed Spark-based feature engineering pipeline for:

1. Championship prediction
2. Race winner prediction
3. Analytics and visualization

This script:
- Loads raw F1 datasets
- Cleans and joins tables
- Computes advanced racing analytics features
- Uses Spark Window Functions heavily
- Saves ML-ready parquet datasets

Usage:
    python src/feature_engineering.py
"""

import os
import sys
import logging
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ERGAST_DATA_DIR = os.path.join(RAW_DATA_DIR, "ergast")
FEATURE_DIR  = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────────────────────


def get_spark():
    spark = SparkSession.builder \
        .appName("F1FeatureEngineering") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────


def load_data(spark):
    log.info("Loading raw datasets...")
    start = time.time()

    def raw_csv_path(filename):
        direct_path = os.path.join(RAW_DATA_DIR, filename)
        ergast_path = os.path.join(ERGAST_DATA_DIR, filename)
        return direct_path if os.path.exists(direct_path) else ergast_path

    races = spark.read.csv(
        raw_csv_path("races.csv"),
        header=True,
        inferSchema=True
    )

    results = spark.read.csv(
        raw_csv_path("results.csv"),
        header=True,
        inferSchema=True
    )

    drivers = spark.read.csv(
        raw_csv_path("drivers.csv"),
        header=True,
        inferSchema=True
    )

    constructors = spark.read.csv(
        raw_csv_path("constructors.csv"),
        header=True,
        inferSchema=True
    )

    qualifying = spark.read.csv(
        raw_csv_path("qualifying.csv"),
        header=True,
        inferSchema=True
    )

    circuits = spark.read.csv(
        raw_csv_path("circuits.csv"),
        header=True,
        inferSchema=True
    )

    status = spark.read.csv(
        raw_csv_path("status.csv"),
        header=True,
        inferSchema=True
    )

    pit_stops_path = raw_csv_path("pit_stops.csv")

    if os.path.exists(pit_stops_path):
        pit_stops = spark.read.csv(
            pit_stops_path,
            header=True,
            inferSchema=True
        )
    else:
        pit_stops = None
        log.warning("pit_stops.csv not found")

    datasets = {
        "races": races,
        "results": results,
        "drivers": drivers,
        "constructors": constructors,
        "qualifying": qualifying,
        "circuits": circuits,
        "status": status,
        "pit_stops": pit_stops,
    }

    log.info(f"Loading stage completed in {time.time() - start:.2f}s")
    return datasets


# ─────────────────────────────────────────────────────────────
# BUILD MASTER DATAFRAME
# ─────────────────────────────────────────────────────────────


def build_master_dataframe(data):
    log.info("Building master dataframe...")
    start = time.time()

    races = data["races"]
    results = data["results"]
    drivers = data["drivers"]
    constructors = data["constructors"]
    qualifying = data["qualifying"]
    circuits = data["circuits"]
    status = data["status"]

    # Basic race information
    race_df = races.select(
        "raceId",
        "year",
        "round",
        "circuitId",
        "name",
        "date"
    )

    # Driver names
    driver_df = drivers.select(
        "driverId",
        F.concat_ws(" ", "forename", "surname").alias("driver_name")
    )

    # Constructor names
    constructor_df = constructors.select(
        "constructorId",
        F.col("name").alias("constructor_name")
    )

    # Circuit names
    circuit_df = circuits.select(
        "circuitId",
        F.col("name").alias("circuit_name")
    )

    # Status table
    status_df = status.select(
        "statusId",
        "status"
    )

    # Qualifying positions
    quali_df = qualifying.select(
        "raceId",
        "driverId",
        F.col("position").alias("quali_position")
    )

    # Main join
    df = results \
        .join(race_df, on="raceId", how="left") \
        .join(driver_df, on="driverId", how="left") \
        .join(constructor_df, on="constructorId", how="left") \
        .join(circuit_df, on="circuitId", how="left") \
        .join(status_df, on="statusId", how="left") \
        .join(quali_df, on=["raceId", "driverId"], how="left")

    log.info(f"Joins stage completed in {time.time() - start:.2f}s")
    return df


# ─────────────────────────────────────────────────────────────
# CLEAN DATA
# ─────────────────────────────────────────────────────────────


def clean_data(df):
    log.info("Cleaning data...")

    df = df.dropDuplicates()

    # Cast important numeric columns
    numeric_cols = [
        "positionOrder",
        "grid",
        "points",
        "laps",
        "fastestLapSpeed",
        "quali_position"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.when(F.col(col).cast("string") == "\\N", None)
                .otherwise(F.col(col))
                .cast("double")
            )

    # Fill nulls
    df = df.fillna({
        "points": 0,
        "grid": 20,
        "quali_position": 20,
        "fastestLapSpeed": 0,
    })

    # Final race position
    df = df.withColumn(
        "final_position",
        F.col("positionOrder").cast("int")
    )

    return df


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────


def create_basic_features(df):
    log.info("Creating basic features...")

    # Win label
    df = df.withColumn(
        "is_winner",
        F.when(F.col("final_position") == 1, 1).otherwise(0)
    )

    # Podium label
    df = df.withColumn(
        "is_podium",
        F.when(F.col("final_position") <= 3, 1).otherwise(0)
    )

    # Positions gained
    df = df.withColumn(
        "positions_gained",
        F.col("grid") - F.col("final_position")
    )

    # DNF flag
    df = df.withColumn(
        "is_dnf",
        F.when(F.col("status") != "Finished", 1).otherwise(0)
    )

    # Wet race indicator
    df = df.withColumn(
        "wet_race",
        F.when(
            F.lower(F.col("circuit_name")).contains("monaco"),
            0
        ).otherwise(0)
    )

    return df


# ─────────────────────────────────────────────────────────────
# WINDOW FEATURES
# ─────────────────────────────────────────────────────────────


def create_window_features(df):
    log.info("Creating Spark window features...")
    start = time.time()

    # Base sequential window
    driver_window = Window.partitionBy(
        "driverId",
        "year"
    ).orderBy("round")

    # Last 3 races
    last3_window = driver_window.rowsBetween(-2, 0)

    # Last 5 races
    last5_window = driver_window.rowsBetween(-4, 0)

    # Full season cumulative window
    cumulative_window = driver_window.rowsBetween(
        Window.unboundedPreceding,
        0
    )

    # Constructor cumulative window
    constructor_window_seq = Window.partitionBy(
        "constructorId",
        "year"
    ).orderBy("round")

    constructor_cumulative_window = constructor_window_seq.rowsBetween(
        Window.unboundedPreceding,
        0
    )

    # Driver momentum
    df = df.withColumn(
        "avg_finish_last_3",
        F.avg("final_position").over(last3_window)
    )

    # Rolling average finish
    df = df.withColumn(
        "rolling_avg_finish",
        F.avg("final_position").over(last5_window)
    )

    # Rolling DNFs
    df = df.withColumn(
        "rolling_dnfs",
        F.sum("is_dnf").over(last5_window)
    )

    # Rolling points last 3 races
    df = df.withColumn(
        "rolling_points_last_3",
        F.avg("points").over(last3_window)
    )

    # Cumulative points
    df = df.withColumn(
        "cumulative_points",
        F.sum("points").over(cumulative_window)
    )

    # Win flag
    df = df.withColumn(
        "win_flag",
        F.when(F.col("final_position") == 1, 1).otherwise(0)
    )

    # Cumulative wins
    df = df.withColumn(
        "cumulative_wins",
        F.sum("win_flag").over(cumulative_window)
    )

    # Cumulative DNFs
    df = df.withColumn(
        "cumulative_dnfs",
        F.sum("is_dnf").over(cumulative_window)
    )

    # Rolling wins last 5 races
    df = df.withColumn(
        "rolling_wins_last_5",
        F.sum("win_flag").over(last5_window)
    )

    # DNF trend
    df = df.withColumn(
        "dnfs_last_5",
        F.sum("is_dnf").over(last5_window)
    )

    # Average qualifying position
    df = df.withColumn(
        "avg_quali_last_3",
        F.avg("quali_position").over(last3_window)
    )

    # Average positions gained
    df = df.withColumn(
        "avg_positions_gained",
        F.avg("positions_gained").over(last5_window)
    )

    # Best finish so far
    df = df.withColumn(
        "best_finish_so_far",
        F.min("final_position").over(cumulative_window)
    )

    # Race count
    df = df.withColumn(
        "races_completed",
        F.row_number().over(driver_window)
    )

    # Momentum score
    df = df.withColumn(
        "momentum_score",
        F.col("rolling_points_last_3") /
        (F.col("rolling_avg_finish") + 1)
    )

    # Finish consistency
    df = df.withColumn(
        "finish_position_stddev",
        F.stddev("final_position").over(last5_window)
    )

    # Consistency score
    df = df.withColumn(
        "consistency_score",
        1 / (F.col("finish_position_stddev") + 1)
    )

    # Podium rate
    df = df.withColumn(
        "podium_rate",
        F.sum("is_podium").over(cumulative_window) /
        F.col("races_completed")
    )

    # Win rate
    df = df.withColumn(
        "win_rate",
        F.col("cumulative_wins") /
        F.col("races_completed")
    )

    # Reliability score
    df = df.withColumn(
        "reliability_score",
        1 - (
            F.col("cumulative_dnfs") /
            F.col("races_completed")
        )
    )

    # Constructor cumulative points
    df = df.withColumn(
        "constructor_cumulative_points",
        F.sum("points").over(constructor_cumulative_window)
    )

    # Career races before this season
    career_window = Window.partitionBy(
        "driverId"
    ).orderBy("date")

    career_history = career_window.rowsBetween(
        Window.unboundedPreceding,
        -1
    )

    df = df.withColumn(
        "career_races",
        F.count("*").over(career_history)
    )

    log.info(f"Window features stage completed in {time.time() - start:.2f}s")
    return df


# ─────────────────────────────────────────────────────────────
# CONSTRUCTOR FEATURES
# ─────────────────────────────────────────────────────────────


def create_constructor_features(df):
    log.info("Creating constructor features...")

    constructor_window = Window.partitionBy(
        "constructorId",
        "year"
    )

    constructor_seq_window = Window.partitionBy(
        "constructorId",
        "year"
    ).orderBy("round")

    constructor_cumulative_window = constructor_seq_window.rowsBetween(
        Window.unboundedPreceding,
        0
    )

    # Constructor average points
    df = df.withColumn(
        "constructor_avg_points",
        F.avg("points").over(constructor_window)
    )

    # Constructor total wins
    df = df.withColumn(
        "constructor_total_wins",
        F.sum("win_flag").over(constructor_window)
    )

    # Constructor podium rate
    df = df.withColumn(
        "constructor_podium_rate",
        F.avg("is_podium").over(constructor_window)
    )

    # Constructor cumulative points
    df = df.withColumn(
        "constructor_cumulative_points",
        F.sum("points").over(constructor_cumulative_window)
    )

    # Constructor rolling points
    df = df.withColumn(
        "constructor_rolling_points",
        F.avg("points").over(
            constructor_seq_window.rowsBetween(-4, 0)
        )
    )

    # Constructor momentum
    df = df.withColumn(
        "constructor_momentum",
        F.col("constructor_rolling_points") /
        (F.col("round") + 1)
    )

    return df


# ─────────────────────────────────────────────────────────────
# CIRCUIT FEATURES
# ─────────────────────────────────────────────────────────────


def create_circuit_features(df):
    log.info("Creating circuit features...")

    circuit_window = Window.partitionBy(
        "driverId",
        "circuitId"
    ).orderBy("date")

    historical_circuit_window = circuit_window.rowsBetween(
        Window.unboundedPreceding,
        -1
    )

    # Historical average finish at circuit
    df = df.withColumn(
        "avg_finish_at_circuit",
        F.avg("final_position").over(historical_circuit_window)
    )

    # Historical wins at circuit
    df = df.withColumn(
        "wins_at_circuit",
        F.sum("win_flag").over(historical_circuit_window)
    )

    return df


# ─────────────────────────────────────────────────────────────
# CHAMPIONSHIP LABELS
# ─────────────────────────────────────────────────────────────


def create_championship_labels(df):
    log.info("Creating championship labels...")

    # Final season standings
    season_window = Window.partitionBy(
        "driverId",
        "year"
    )

    df = df.withColumn(
        "season_total_points",
        F.max("cumulative_points").over(season_window)
    )

    # Rank drivers within each season
    championship_window = Window.partitionBy("year") \
        .orderBy(F.desc("season_total_points"))

    df = df.withColumn(
        "season_rank",
        F.dense_rank().over(championship_window)
    )

    # Champion label
    df = df.withColumn(
        "is_champion",
        F.when(F.col("season_rank") == 1, 1).otherwise(0)
    )

    return df


# ─────────────────────────────────────────────────────────────
# RACES REMAINING
# ─────────────────────────────────────────────────────────────


def create_remaining_races_feature(df):
    log.info("Creating races remaining feature...")

    total_races_df = df.groupBy("year") \
        .agg(F.max("round").alias("total_races"))

    df = df.join(total_races_df, on="year", how="left")

    df = df.withColumn(
        "races_remaining",
        F.col("total_races") - F.col("round")
    )

    return df


# ─────────────────────────────────────────────────────────────
# GAP TO LEADER
# ─────────────────────────────────────────────────────────────


def create_gap_to_leader(df):
    log.info("Creating gap to championship leader feature...")

    round_window = Window.partitionBy(
        "year",
        "round"
    )

    df = df.withColumn(
        "leader_points",
        F.max("cumulative_points").over(round_window)
    )

    df = df.withColumn(
        "gap_to_leader",
        F.col("leader_points") - F.col("cumulative_points")
    )

    return df


# ─────────────────────────────────────────────────────────────
# SAVE DATASETS
# ─────────────────────────────────────────────────────────────


def generate_feature_statistics(df):
    log.info("Generating feature statistics...")

    stats = df.select([
        F.mean(c).alias(f"{c}_mean")
        for c, t in df.dtypes
        if t in ("double", "int", "bigint", "float")
    ])

    stats.show(truncate=False)


def save_feature_datasets(df):
    log.info("Saving feature datasets...")
    start = time.time()

    df = df.cache()
    df.count()

    df = df.withColumn(
        "driver_id",
        F.col("driverId").cast("string")
    ).withColumn(
        "won_championship",
        F.col("is_champion")
    )

    # Championship features
    champ_cols = [
        "driver_id",
        "driverId",
        "driver_name",
        "constructor_name",
        "year",
        "round",
        "circuit_name",
        "final_position",
        "points",
        "cumulative_points",
        "cumulative_wins",
        "gap_to_leader",
        "races_remaining",
        "rolling_avg_finish",
        "rolling_dnfs",
        "rolling_points_last_3",
        "rolling_wins_last_5",
        "momentum_score",
        "consistency_score",
        "podium_rate",
        "win_rate",
        "reliability_score",
        "avg_finish_last_3",
        "positions_gained",
        "dnfs_last_5",
        "constructor_avg_points",
        "constructor_cumulative_points",
        "constructor_rolling_points",
        "constructor_momentum",
        "avg_finish_at_circuit",
        "career_races",
        "is_champion",
        "won_championship"
    ]

    existing_champ_cols = [c for c in champ_cols if c in df.columns]

    champ_df = df.select(existing_champ_cols)
    champ_df = champ_df.repartition(8)

    champ_output = os.path.join(
        FEATURE_DIR,
        "champ_features.parquet"
    )

    champ_df.write.mode("overwrite").parquet(champ_output)

    log.info(f"Saved championship features → {champ_output}")

    # Race winner features
    race_cols = [
        "driverId",
        "driver_name",
        "constructor_name",
        "year",
        "round",
        "circuit_name",
        "grid",
        "quali_position",
        "final_position",
        "points",
        "positions_gained",
        "avg_finish_last_3",
        "rolling_avg_finish",
        "constructor_avg_points",
        "avg_finish_at_circuit",
        "dnfs_last_5",
        "is_winner"
    ]

    existing_race_cols = [c for c in race_cols if c in df.columns]

    race_df = df.select(existing_race_cols)
    race_df = race_df.repartition(8)

    race_output = os.path.join(
        FEATURE_DIR,
        "race_features.parquet"
    )

    race_df.write.mode("overwrite").parquet(race_output)

    log.info(f"Saved race features → {race_output}")


# ─────────────────────────────────────────────────────────────
    log.info(f"Parquet writes stage completed in {time.time() - start:.2f}s")


# ANALYTICS TABLES
# ─────────────────────────────────────────────────────────────


def generate_analytics_tables(df):
    log.info("Generating analytics tables...")
    start = time.time()

    # Constructor dominance
    constructor_stats = df.groupBy(
        "constructor_name"
    ).agg(
        F.avg("points").alias("avg_points"),
        F.sum("win_flag").alias("total_wins"),
        F.avg("is_podium").alias("podium_rate")
    ).orderBy(F.desc("avg_points"))

    constructor_output = os.path.join(
        RESULTS_DIR,
        "constructor_dominance.parquet"
    )

    constructor_stats.write.mode("overwrite").parquet(constructor_output)

    # Driver win rates
    driver_stats = df.groupBy(
        "driver_name"
    ).agg(
        F.sum("win_flag").alias("wins"),
        F.avg("is_winner").alias("win_rate"),
        F.avg("points").alias("avg_points")
    ).orderBy(F.desc("wins"))

    driver_output = os.path.join(
        RESULTS_DIR,
        "driver_statistics.parquet"
    )

    driver_stats.write.mode("overwrite").parquet(driver_output)

    # Circuit difficulty
    circuit_stats = df.groupBy(
        "circuit_name"
    ).agg(
        F.avg("is_dnf").alias("avg_dnf_rate"),
        F.avg("positions_gained").alias("avg_overtakes")
    )

    circuit_output = os.path.join(
        RESULTS_DIR,
        "circuit_statistics.parquet"
    )

    circuit_stats.write.mode("overwrite").parquet(circuit_output)

    log.info("Analytics tables saved.")
    log.info(f"Analytics generation stage completed in {time.time() - start:.2f}s")

    return constructor_stats, driver_stats, circuit_stats


def run():
    spark = get_spark()

    try:
        data = load_data(spark)
        df = build_master_dataframe(data)
        df = clean_data(df)
        df = create_basic_features(df)
        df = create_window_features(df)
        df = create_constructor_features(df)
        df = create_circuit_features(df)
        df = create_championship_labels(df)
        df = create_remaining_races_feature(df)
        df = create_gap_to_leader(df)

        generate_feature_statistics(df)
        save_feature_datasets(df)
        generate_analytics_tables(df)

        log.info("Final dataset partitions:")
        log.info(df.rdd.getNumPartitions())
        log.info("=== Feature engineering complete ===")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()


