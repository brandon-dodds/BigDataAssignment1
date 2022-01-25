from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv")
    splits = df.randomSplit([0.7, 0.3])
    print(splits[0].count())
    print(splits[1].count())
    string_indexer = StringIndexer()
    si_model = string_indexer.fit(splits[0])
    td = si_model.transform(df)
    dt = DecisionTreeClassifier(maxDepth=2)
    model = dt.fit(td)
    print(model.getLabelCol())


main()
