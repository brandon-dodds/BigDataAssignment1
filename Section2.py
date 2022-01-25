from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorIndexer


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv")
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)
    feature_indexer.fit(df)


main()
