from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, max, min, variance


def get_features(dataframe):
    data_mean = dataframe.select(*[mean(c).alias(c) for c in dataframe.columns[1:]])
    data_max = dataframe.select(*[max(c).alias(c) for c in dataframe.columns[1:]])
    data_min = dataframe.select(*[min(c).alias(c) for c in dataframe.columns[1:]])
    data_variance = dataframe.select(*[variance(c).alias(c) for c in dataframe.columns[1:]])
    data_median = dataframe.approxQuantile([c for c in dataframe.columns[1:]], [0.5], 0.25)
    data_mode = [dataframe.groupby(i).count().orderBy("count", ascending=False).first()[0] for i in dataframe.columns]
    return data_min, data_max, data_mean, data_variance, data_median, data_mode


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df_normal = df.filter(df.Status == "Normal")
    df_abnormal = df.filter(df.Status == "Abnormal")

    normal_min, normal_max, normal_mean, normal_variance, normal_median, normal_mode = get_features(df_normal)
    abnormal_min, abnormal_max, abnormal_mean, abnormal_variance, abnormal_median, normal_mode = \
        get_features(df_abnormal)
    # names = df.schema.names
    #
    # for name in names[1:]:
    #     df.filter(df.Status == "Normal").select(mean(name)).show()


main()
