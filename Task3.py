from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sn
import matplotlib.pyplot as plt


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

    df_normal = df.filter(df.Status == "Abnormal")
    # df_abnormal = df.filter(df.Status == "Abnormal")

    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=df_normal.columns[1:], outputCol=vector_col)
    df_vector = assembler.transform(df_normal).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col)
    sn.heatmap(matrix.collect()[0]["pearson({})".format(vector_col)].values.reshape(12, 12), annot=True)
    plt.show()


main()
