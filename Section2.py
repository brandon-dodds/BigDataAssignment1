from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    label_indexer = StringIndexer(inputCol=df.columns[0], outputCol='indexedLabel').fit(df)
    feature_assembler = VectorAssembler(inputCols=df.columns[1:],
                                        outputCol='assembledFeatures').transform(df)
    feature_indexer = VectorIndexer(inputCol='assembledFeatures', outputCol='indexedFeatures', maxCategories=4).fit(df)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    dt = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures')
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])
    model = pipeline.fit(trainingData)


main()
