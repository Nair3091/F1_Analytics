"""
race_winner_model.py
--------------------
Spark MLlib RandomForest model for Formula 1 race winner prediction.

Usage:
    python src/race_winner_model.py
"""

import logging
import os
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "race_winner_random_forest")

os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


FEATURE_COLS = [
    "grid",
    "quali_position",
    "rolling_avg_finish",
    "avg_finish_last_3",
    "constructor_avg_points",
    "avg_finish_at_circuit",
    "dnfs_last_5",
    "positions_gained",
]

TARGET_COL = "is_winner"


def get_spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.builder \
        .appName("F1RaceWinnerRandomForest") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_race_features(spark):
    path = os.path.join(FEATURE_DIR, "race_features.parquet")
    log.info(f"Loading race features from {path}")

    df = spark.read.parquet(path)
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    select_cols = ["year", "round", "driver_name", "constructor_name"] + FEATURE_COLS + [TARGET_COL]
    existing_cols = [c for c in select_cols if c in df.columns]

    df = df.select(existing_cols)
    for col in FEATURE_COLS + [TARGET_COL]:
        df = df.withColumn(col, F.col(col).cast("double"))

    df = df.filter(F.col(TARGET_COL).isNotNull())
    log.info(f"Loaded {df.count()} race-driver rows")
    log.info(f"Winners in dataset: {df.filter(F.col(TARGET_COL) == 1).count()}")
    return df


def train_test_split(df):
    train_df = df.filter(F.col("year") <= 2018)
    test_df = df.filter(F.col("year") > 2018)

    if test_df.count() == 0:
        log.warning("No post-2018 test rows found; using deterministic random split instead.")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    log.info(f"Train rows: {train_df.count()}")
    log.info(f"Test rows: {test_df.count()}")
    return train_df, test_df


def build_pipeline():
    imputed_cols = [f"{c}_imputed" for c in FEATURE_COLS]

    imputer = Imputer(
        inputCols=FEATURE_COLS,
        outputCols=imputed_cols
    ).setStrategy("median")

    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="features"
    )

    rf = RandomForestClassifier(
        labelCol=TARGET_COL,
        featuresCol="features",
        numTrees=200,
        maxDepth=8,
        minInstancesPerNode=5,
        subsamplingRate=0.8,
        seed=42
    )

    return Pipeline(stages=[imputer, assembler, rf])


def evaluate_model(model, test_df):
    predictions = model.transform(test_df)

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=TARGET_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol=TARGET_COL,
        predictionCol="prediction",
        metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=TARGET_COL,
        predictionCol="prediction",
        metricName="f1"
    )

    auc = auc_evaluator.evaluate(predictions)
    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)

    log.info(f"Race winner ROC-AUC: {auc:.4f}")
    log.info(f"Race winner accuracy: {accuracy:.4f}")
    log.info(f"Race winner F1: {f1:.4f}")

    top_predictions = predictions.select(
        "year",
        "round",
        "driver_name",
        "constructor_name",
        TARGET_COL,
        "prediction",
        vector_to_array("probability").getItem(1).alias("winner_probability")
    ).orderBy(F.desc("winner_probability"))

    output_path = os.path.join(RESULTS_DIR, "race_winner_predictions.parquet")
    top_predictions.write.mode("overwrite").parquet(output_path)
    log.info(f"Saved race winner predictions to {output_path}")

    log.info("Top 10 predicted race-winning probabilities:")
    top_predictions.show(10, truncate=False)

    return auc, accuracy, f1


def save_model(model):
    model.write().overwrite().save(MODEL_DIR)
    log.info(f"Saved RandomForest model to {MODEL_DIR}")


def run():
    spark = get_spark()

    try:
        df = load_race_features(spark)
        train_df, test_df = train_test_split(df)

        pipeline = build_pipeline()
        log.info("Training Spark MLlib RandomForest race winner model...")
        model = pipeline.fit(train_df)

        evaluate_model(model, test_df)
        save_model(model)
        log.info("=== Race winner model complete ===")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
