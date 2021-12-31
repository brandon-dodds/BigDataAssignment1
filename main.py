from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, max, min, variance


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    # names = df.schema.names
    #
    # for name in names[1:]:
    #     df.filter(df.Status == "Normal").select(mean(name)).show()

    data_mean = df.select(*[mean(c).alias(c) for c in df.columns[1:]])
    data_max = df.select(*[max(c).alias(c) for c in df.columns[1:]])
    data_min = df.select(*[min(c).alias(c) for c in df.columns[1:]])
    data_variance = df.select(*[variance(c).alias(c) for c in df.columns[1:]])
    data_median = df.approxQuantile([c for c in df.columns[1:]], [0.5], 0.25)
    print(data_median)


main()
