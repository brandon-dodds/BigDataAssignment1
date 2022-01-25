from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sn
import matplotlib.pyplot as plt


def create_matrix(dataset, x_label):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=dataset.columns[1:], outputCol=vector_col)
    df_vector = assembler.transform(dataset).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col)
    sn.heatmap(matrix.collect()[0]["pearson({})".format(vector_col)].values.reshape(12, 12), annot=True)
    plt.xlabel(x_label)
    plt.show()


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df_normal = df.filter(df.Status == "Normal")
    df_abnormal = df.filter(df.Status == "Abnormal")
    create_matrix(df_normal, "Normal")
    create_matrix(df_abnormal, "Abnormal")


main()
