import pandas as pd

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("PackageLvlFeatureCreator") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

df= spark.read.csv('')