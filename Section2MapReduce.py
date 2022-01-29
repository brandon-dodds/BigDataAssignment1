from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_big_dataset.csv", inferSchema=True, header=True)
    df.show()


main()
