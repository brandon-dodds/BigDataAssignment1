from pyspark.sql import SparkSession
from statistics import multimode
import csv
from matplotlib import pyplot as plt


def return_modes(column):
    column_vals = column.collect()
    return multimode(column_vals)


def prepare_summary_statistics(column):
    table_data = []
    # column_summary.show()
    col_mode = [x[0] for x in return_modes(column)]
    rows = column.summary("min", "max", "mean", "50%", "stddev").collect()
    return_modes(column)
    col_min, col_max, col_mean, col_median, col_variance = rows[0][1], rows[1][1], rows[2][1], rows[3][1], rows[4][1]
    return col_min, col_max, col_mean, col_median, col_mode, float(col_variance) ** 2


def main():
    # Load pyspark dataframe
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df_normal = df.filter(df.Status == "Normal")
    df_abnormal = df.filter(df.Status == "Abnormal")
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['type', 'name', 'minimum', 'maximum', 'mean', 'median', 'modes', 'variance'])
        for c in df.columns[1:]:
            df_normal_col = df_normal.select(c)
            df_abnormal_col = df_abnormal.select(c)
            col_min, col_max, col_mean, col_median, col_mode, col_variance = prepare_summary_statistics(df_normal_col)
            writer.writerow(['normal', c, col_min, col_max, col_mean, col_median, col_mode, col_variance])
            col_min, col_max, col_mean, col_median, col_mode, col_variance = prepare_summary_statistics(df_abnormal_col)
            writer.writerow(['abnormal', c, col_min, col_max, col_mean, col_median, col_mode, col_variance])
    df_normal.toPandas().boxplot()
    plt.xlabel("normal")
    plt.show()
    df_abnormal.toPandas().boxplot()
    plt.xlabel("abnormal")
    plt.show()


main()
