from pyspark.sql import SparkSession


def prepare_summary_statistics(column):
    table_data = []
    # column_summary.show()
    rows = column.summary().collect()
    col_min, col_max, col_mean, col_median, col_variance = rows[3][1], rows[7][1], rows[1][1], rows[5][1], rows[2][1]
    table_data.extend([col_min, col_max, col_mean, col_median, col_variance])
    return table_data


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df_normal = df.filter(df.Status == "Normal")
    df_abnormal = df.filter(df.Status == "Abnormal")

    for c in df.columns[1:]:
        df_col = df_normal.select(c)
        prepare_summary_statistics(df_col)


main()
