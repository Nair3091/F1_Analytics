# inspect_features.py

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Inspect").getOrCreate()

df = spark.read.parquet("data/features/champ_features.parquet")

df.printSchema()
df.show(10)

print("Rows:", df.count())
print("Columns:", len(df.columns))