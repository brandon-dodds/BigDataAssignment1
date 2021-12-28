import findspark
from pyspark.sql import SparkSession


def main():
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql("select 'spark' as hello")
    df.show()


main()
