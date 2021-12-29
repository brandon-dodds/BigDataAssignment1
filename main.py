from pyspark.sql import SparkSession


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    df.show()
    df.printSchema()


main()
