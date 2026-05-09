"""
features.py
───────────
Builds feature tables for both prediction tasks:
  1. race_features.parquet  — one row per driver per race (for race winner model)
  2. champ_features.parquet — one row per driver per round in season (for championship model)

Usage:
    python src/features.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StringType, IntegerType, FloatType
)

from config import (
    ERGAST_RAW, OPENF1_RAW,
    DATA_PROCESSED, DATA_FEATURES,
    SPARK_APP_NAME,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Spark Session ──────────────────────────────────────────────────────────

def get_spark():
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["HADOOP_HOME"]           = "C:\\Hadoop"

    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.files.maxPartitionBytes", "134217728") \
        .config("spark.default.parallelism", "8") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ── Load Ergast CSVs ───────────────────────────────────────────────────────

def load_ergast(spark):
    log.info("Loading Ergast CSVs...")

    def read(filename):
        path = os.path.join(ERGAST_RAW, filename)
        return spark.read.csv(path, header=True, inferSchema=True, nullValue="\\N")

    races        = read("races.csv")
    results      = read("results.csv")
    drivers      = read("drivers.csv")
    constructors = read("constructors.csv")
    qualifying   = read("qualifying.csv")
    pit_stops    = read("pit_stops.csv")
    standings    = read("driver_standings.csv")
    circuits     = read("circuits.csv")
    status       = read("status.csv")

    log.info("Ergast CSVs loaded.")
    return races, results, drivers, constructors, qualifying, pit_stops, standings, circuits, status


# ── Load OpenF1 JSONs ──────────────────────────────────────────────────────

def collect_json_paths(base_dir, filename):
    """
    Walks the openf1 raw directory and collects paths to all non-empty
    JSON files matching the given filename.
    """
    paths = []
    for year in os.listdir(base_dir):
        year_dir = os.path.join(base_dir, year)
        if not os.path.isdir(year_dir):
            continue
        for session in os.listdir(year_dir):
            session_dir = os.path.join(year_dir, session)
            if not os.path.isdir(session_dir):
                continue
            file_path = os.path.join(session_dir, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 10:
                paths.append(file_path)
    return paths


def load_openf1_endpoint(spark, endpoint):
    """
    Reads all JSON files for a given endpoint using Spark's native JSON reader.
    Skips empty files using file size check — avoids triggering any Spark actions.
    """
    paths = collect_json_paths(OPENF1_RAW, f"{endpoint}.json")

    if not paths:
        log.warning(f"No files found for OpenF1 endpoint: {endpoint}")
        return None

    log.info(f"Reading {len(paths)} files for endpoint: {endpoint}")

    df = spark.read \
        .option("multiLine", "true") \
        .json(paths)

    log.info(f"Loaded OpenF1 {endpoint}: schema has {len(df.columns)} columns")
    return df


def load_openf1_sessions(spark):
    """Loads all session metadata JSONs."""
    paths = collect_json_paths(OPENF1_RAW, "session_meta.json")

    if not paths:
        log.warning("No session metadata files found.")
        return None

    log.info(f"Reading {len(paths)} session metadata files.")
    return spark.read.option("multiLine", "true").json(paths)


def load_openf1_quali_laps(spark):
    """Loads qualifying lap data for grid position features."""
    paths = []
    for year in os.listdir(OPENF1_RAW):
        year_dir = os.path.join(OPENF1_RAW, year)
        if not os.path.isdir(year_dir):
            continue
        for session in os.listdir(year_dir):
            if not session.startswith("quali_"):
                continue
            file_path = os.path.join(year_dir, session, "laps.json")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 10:
                paths.append(file_path)

    if not paths:
        log.warning("No qualifying lap files found.")
        return None

    log.info(f"Reading {len(paths)} qualifying lap files.")
    return spark.read.option("multiLine", "true").json(paths)


# ── Ergast Feature Building ────────────────────────────────────────────────

def build_ergast_features(spark, races, results, drivers, constructors,
                           qualifying, pit_stops, standings, circuits, status):
    log.info("Building Ergast features...")

    drivers = drivers.withColumn(
        "driver_name",
        F.concat(F.col("forename"), F.lit(" "), F.col("surname"))
    )

    base = results \
        .join(races.select("raceId", "year", "round", "circuitId", "name", "date"),
              "raceId") \
        .join(drivers.select("driverId", "driver_name", "nationality"),
              "driverId") \
        .join(constructors.select("constructorId", "name").withColumnRenamed("name", "team"),
              "constructorId") \
        .join(circuits.select("circuitId", "country"), "circuitId") \
        .join(status.select("statusId", "status"), "statusId")

    base = base \
        .withColumn("final_position", F.col("positionOrder").cast(IntegerType())) \
        .withColumn("points",         F.col("points").cast(FloatType())) \
        .withColumn("grid",           F.col("grid").cast(IntegerType())) \
        .withColumn("year",           F.col("year").cast(IntegerType())) \
        .withColumn("round",          F.col("round").cast(IntegerType())) \
        .withColumn("date",           F.to_date(F.col("date")))

    base = base.withColumn("dnf",
        F.when(F.col("status").isin(
            ["Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps"]
        ), 0).otherwise(1)
    )

    # Rolling driver form — last 5 races
    driver_win = Window.partitionBy("driverId").orderBy("date").rowsBetween(-5, -1)

    base = base \
        .withColumn("rolling_avg_finish",
            F.avg("final_position").over(driver_win)) \
        .withColumn("rolling_points",
            F.sum("points").over(driver_win)) \
        .withColumn("rolling_wins",
            F.sum(F.when(F.col("final_position") == 1, 1).otherwise(0)).over(driver_win)) \
        .withColumn("rolling_dnfs",
            F.sum(F.col("dnf")).over(driver_win))

    # Constructor rolling points
    constructor_win = Window.partitionBy("constructorId").orderBy("date").rowsBetween(-5, -1)
    base = base.withColumn("constructor_rolling_points",
        F.sum("points").over(constructor_win))

    # Driver circuit history
    circuit_win = Window.partitionBy("driverId", "circuitId") \
                        .orderBy("date") \
                        .rowsBetween(Window.unboundedPreceding, -1)
    base = base \
        .withColumn("driver_circuit_starts",
            F.count("raceId").over(circuit_win)) \
        .withColumn("driver_circuit_wins",
            F.sum(F.when(F.col("final_position") == 1, 1).otherwise(0)).over(circuit_win))

    # Qualifying / grid position
    quali_clean = qualifying \
        .withColumn("grid_position",
            F.coalesce(F.col("position").cast(IntegerType()), F.lit(20))) \
        .select("raceId", "driverId", "grid_position",
                F.col("q1").alias("q1_time"),
                F.col("q2").alias("q2_time"),
                F.col("q3").alias("q3_time"))

    base = base.join(quali_clean, ["raceId", "driverId"], how="left")
    base = base.withColumn("grid_position",
        F.coalesce(F.col("grid_position"), F.col("grid"), F.lit(20)))

    pole_win = Window.partitionBy("raceId")
    base = base.withColumn("gap_to_pole",
        F.col("grid_position") - F.min("grid_position").over(pole_win))

    # Pit stop features
    pit_agg = pit_stops \
        .withColumn("duration_ms", F.col("milliseconds").cast(FloatType())) \
        .groupBy("raceId", "driverId") \
        .agg(
            F.count("stop").alias("num_pit_stops"),
            F.avg("duration_ms").alias("avg_pit_ms"),
            F.min("lap").alias("first_pit_lap")
        )

    base = base.join(pit_agg, ["raceId", "driverId"], how="left") \
        .fillna({"num_pit_stops": 0, "avg_pit_ms": 0.0, "first_pit_lap": 0})

    # Target labels
    base = base \
        .withColumn("is_winner",
            F.when(F.col("final_position") == 1, 1).otherwise(0)) \
        .withColumn("is_podium",
            F.when(F.col("final_position") <= 3, 1).otherwise(0))

    log.info("Ergast features built.")
    return base


# ── OpenF1 Feature Building ────────────────────────────────────────────────

def build_openf1_features(spark):
    log.info("Building OpenF1 features...")

    sessions   = load_openf1_sessions(spark)
    laps       = load_openf1_endpoint(spark, "laps")
    stints     = load_openf1_endpoint(spark, "stints")
    pits       = load_openf1_endpoint(spark, "pit")
    weather    = load_openf1_endpoint(spark, "weather")
    rc         = load_openf1_endpoint(spark, "race_control")
    quali_laps = load_openf1_quali_laps(spark)

    if sessions is None or laps is None:
        log.error("Missing critical OpenF1 data — sessions or laps not found.")
        return None

    # Weather per session
    weather_agg = weather \
        .withColumn("rain_flag",
            F.when(F.col("rainfall").cast(FloatType()) > 0, 1).otherwise(0)) \
        .groupBy("session_key") \
        .agg(
            F.avg(F.col("air_temperature").cast(FloatType())).alias("avg_air_temp"),
            F.avg(F.col("track_temperature").cast(FloatType())).alias("avg_track_temp"),
            F.max("rain_flag").alias("rain_flag"),
            F.avg(F.col("wind_speed").cast(FloatType())).alias("avg_wind_speed")
        ) if weather is not None else None

    # Safety car count per session
    sc_agg = rc \
        .filter(F.col("category").isin(["SafetyCar", "VirtualSafetyCar"])) \
        .groupBy("session_key") \
        .agg(F.count("*").alias("safety_car_count")) \
        if rc is not None else None

    # Stint features
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
    stint_agg = stints \
        .groupBy("session_key", "driver_number") \
        .agg(
            F.count("stint_number").alias("num_stints"),
            F.first("compound").alias("starting_compound"),
            F.avg((F.col("lap_end").cast(FloatType()) - F.col("lap_start").cast(FloatType()))).alias("avg_stint_length")
        ) \
        .withColumn("starting_compound_enc",
            F.coalesce(
                F.create_map(*[
                    item for pair in
                    [(F.lit(k), F.lit(v)) for k, v in compound_map.items()]
                    for item in pair
                ])[F.upper(F.col("starting_compound"))],
                F.lit(0)
            )
        ) if stints is not None else None

    # Pit features
    pit_agg = pits \
        .withColumn("pit_duration", F.col("pit_duration").cast(FloatType())) \
        .groupBy("session_key", "driver_number") \
        .agg(
            F.count("*").alias("num_pit_stops_f1"),
            F.avg("pit_duration").alias("avg_pit_duration_s")
        ) if pits is not None else None

    # Qualifying grid features
    if quali_laps is not None:
        quali_best = quali_laps \
            .withColumn("lap_duration", F.col("lap_duration").cast(FloatType())) \
            .filter(F.col("lap_duration").isNotNull()) \
            .groupBy("session_key", "driver_number") \
            .agg(F.min("lap_duration").alias("best_quali_lap"))

        quali_rank_win = Window.partitionBy("session_key").orderBy("best_quali_lap")
        quali_best = quali_best \
            .withColumn("grid_position_f1", F.rank().over(quali_rank_win)) \
            .withColumn("pole_time",
                F.min("best_quali_lap").over(Window.partitionBy("session_key"))) \
            .withColumn("gap_to_pole_f1",
                F.col("best_quali_lap") - F.col("pole_time"))
    else:
        quali_best = None

    # Build base — race sessions joined with per-driver data
    race_sessions = sessions.filter(F.col("session_type") == "Race") \
        .select("session_key", "meeting_key", "year",
                F.col("session_name").alias("race_name"),
                F.col("location").alias("circuit_location"))
    
    

    driver_sessions = laps \
        .select("session_key", "driver_number") \
        .distinct()

    base = race_sessions.join(driver_sessions, "session_key", "inner")

    if weather_agg is not None:
        base = base.join(weather_agg, "session_key", "left")
    if sc_agg is not None:
        base = base.join(sc_agg, "session_key", "left")
    if stint_agg is not None:
        base = base.join(stint_agg, ["session_key", "driver_number"], "left")
    if pit_agg is not None:
        base = base.join(pit_agg, ["session_key", "driver_number"], "left")

    if quali_best is not None:
        quali_sessions = sessions.filter(F.col("session_type") == "Qualifying") \
            .select("session_key", "meeting_key") \
            .withColumnRenamed("session_key", "quali_session_key")

        quali_features = quali_sessions \
            .join(quali_best,
                  F.col("quali_session_key") == quali_best["session_key"], "left") \
            .drop("session_key") \
            .select("meeting_key", "driver_number",
                    "grid_position_f1", "gap_to_pole_f1", "best_quali_lap")

        base = base.join(quali_features, ["meeting_key", "driver_number"], "left")

    base = base.fillna({"safety_car_count": 0, "rain_flag": 0})

    log.info("OpenF1 features built.")
    return base


# ── Unify Ergast + OpenF1 ──────────────────────────────────────────────────

def build_race_features(ergast_features, openf1_features, spark):
    log.info("Building unified race features table...")

    ergast_out = ergast_features.select(
        F.col("raceId").cast(StringType()).alias("race_id"),
        F.col("driverId").cast(StringType()).alias("driver_id"),
        F.col("driver_name"),
        F.col("constructorId").cast(StringType()).alias("constructor_id"),
        F.col("team"),
        F.col("circuitId").cast(StringType()).alias("circuit_id"),
        F.col("year"),
        F.col("round"),
        F.col("date"),
        F.col("grid_position"),
        F.col("gap_to_pole"),
        F.col("final_position"),
        F.col("points"),
        F.col("is_winner"),
        F.col("is_podium"),
        F.col("dnf"),
        F.col("rolling_avg_finish"),
        F.col("rolling_points"),
        F.col("rolling_wins"),
        F.col("rolling_dnfs"),
        F.col("constructor_rolling_points"),
        F.col("driver_circuit_starts"),
        F.col("driver_circuit_wins"),
        F.col("num_pit_stops"),
        F.col("avg_pit_ms"),
        F.lit(None).cast(FloatType()).alias("rain_flag"),
        F.lit(None).cast(FloatType()).alias("avg_track_temp"),
        F.lit(None).cast(IntegerType()).alias("safety_car_count"),
        F.lit(None).cast(IntegerType()).alias("num_stints"),
        F.lit(None).cast(IntegerType()).alias("starting_compound_enc"),
        F.lit("ergast").alias("source")
    )

    if openf1_features is not None:
        openf1_path = os.path.join(DATA_PROCESSED, "openf1_features.parquet")
        openf1_features.write.mode("overwrite").parquet(openf1_path)
        log.info(f"OpenF1 intermediate saved to {openf1_path}")

    log.info("Race features table ready.")
    return ergast_out


# ── Championship Features ──────────────────────────────────────────────────

def build_championship_features(race_features):
    log.info("Building championship features...")

    season_win = Window.partitionBy("driver_id", "year") \
                       .orderBy("round") \
                       .rowsBetween(Window.unboundedPreceding, 0)

    champ = race_features \
        .withColumn("cumulative_points",
            F.sum("points").over(season_win)) \
        .withColumn("cumulative_wins",
            F.sum("is_winner").over(season_win)) \
        .withColumn("cumulative_dnfs",
            F.sum("dnf").over(season_win)) \
        .withColumn("cumulative_podiums",
            F.sum("is_podium").over(season_win))

    leader_win = Window.partitionBy("year", "round")
    champ = champ \
        .withColumn("leader_points",
            F.max("cumulative_points").over(leader_win)) \
        .withColumn("gap_to_leader",
            F.col("leader_points") - F.col("cumulative_points"))

    season_rounds = race_features \
        .groupBy("year") \
        .agg(F.max("round").alias("total_rounds"))

    champ = champ.join(season_rounds, "year") \
        .withColumn("races_remaining",
            F.col("total_rounds") - F.col("round"))

    # Ground truth: championship winner per year
    champion = race_features \
        .groupBy("driver_id", "year") \
        .agg(F.sum("points").alias("season_points")) \
        .withColumn("season_rank",
            F.rank().over(
                Window.partitionBy("year").orderBy(F.desc("season_points")))) \
        .filter(F.col("season_rank") == 1) \
        .select("driver_id", "year", F.lit(1).alias("won_championship"))

    champ = champ.join(champion, ["driver_id", "year"], "left") \
        .fillna({"won_championship": 0})

    log.info("Championship features built.")
    return champ


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(DATA_FEATURES, exist_ok=True)

    spark = get_spark()

    # Load Ergast
    races, results, drivers, constructors, qualifying, \
        pit_stops, standings, circuits, status = load_ergast(spark)

    # Build Ergast features
    ergast_features = build_ergast_features(
        spark, races, results, drivers, constructors,
        qualifying, pit_stops, standings, circuits, status
    )

    # Build OpenF1 features
    openf1_features = build_openf1_features(spark)

    # Unify
    race_features = build_race_features(ergast_features, openf1_features, spark)

    # Fill nulls
    race_features = race_features.fillna({
        "rolling_avg_finish":         10.0,
        "rolling_points":              0.0,
        "rolling_wins":                0,
        "rolling_dnfs":                0,
        "constructor_rolling_points":  0.0,
        "driver_circuit_starts":       0,
        "driver_circuit_wins":         0,
        "gap_to_pole":                 0,
        "num_pit_stops":               0,
        "avg_pit_ms":                  0.0,
        "rain_flag":                   0.0,
        "avg_track_temp":             30.0,
        "safety_car_count":            0,
        "num_stints":                  2,
        "starting_compound_enc":       0,
    })

    # Save race features
    race_path = os.path.join(DATA_FEATURES, "race_features.parquet")
    race_features.write.mode("overwrite").parquet(race_path)
    log.info(f"Race features saved: {race_path}")
    log.info(f"Race features row count: {race_features.count()}")

    # Build and save championship features
    champ_features = build_championship_features(race_features)
    champ_path = os.path.join(DATA_FEATURES, "champ_features.parquet")
    champ_features.write.mode("overwrite").parquet(champ_path)
    log.info(f"Championship features saved: {champ_path}")
    log.info(f"Championship features row count: {champ_features.count()}")

    spark.stop()
    log.info("=== Feature engineering complete ===")


if __name__ == "__main__":
    run()