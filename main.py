from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, max, min, variance


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df.groupby("Status").count().show()


main()
