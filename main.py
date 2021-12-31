from pyspark.sql import SparkSession


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df_normal = df.filter(df.Status == "Normal")
    df_abnormal = df.filter(df.Status == "Abnormal")

    normal_features = df_normal.summary()
    abnormal_features = df_abnormal.summary()
    print(normal_features.show())


main()
