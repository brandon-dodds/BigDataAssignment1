from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sn


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=df.columns[1:], outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col)
    sn.heatmap(matrix.collect()[0]["pearson({})".format(vector_col)].values, annot=True)


main()
