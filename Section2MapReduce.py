from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_big_dataset.csv", inferSchema=True, header=True)
    df = df.drop('Status')
    big_dataset_rdd = df.rdd

    column_names = df.columns
    rows = df.count()

    enumerated_rows = big_dataset_rdd.flatMap(lambda row: enumerate(row))
    reduced_rows_min = enumerated_rows.reduceByKey(lambda data1, data2: min(data1, data2)).map(
        lambda column: (column_names[column[0]], column[1])).collect()
    reduced_rows_max = enumerated_rows.reduceByKey(lambda data1, data2: max(data1, data2)).map(
        lambda column: (column_names[column[0]], column[1])).collect()

    reduced_rows_mean = enumerated_rows.reduceByKey(lambda data1, data2: (data1 + data2)).mapValues(
        lambda data: data / rows).map(lambda column: (column_names[column[0]], column[1])).collect()

    print(reduced_rows_min)
    print(reduced_rows_max)
    print(reduced_rows_mean)


main()
