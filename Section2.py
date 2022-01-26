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
    feature_assembler = VectorAssembler(inputCols=df.columns[1:], outputCol="features")
    feature_data = feature_assembler.transform(df)

    df_train, df_test = feature_data.randomSplit([0.7, 0.3])

    label_indexer = StringIndexer(inputCol=df.columns[0], outputCol="indexedLabel").fit(feature_data)
    feature_indexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(feature_data)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=[4, 5, 4, 3], labelCol="indexedLabel")
    lsvc = LinearSVC(maxIter=1000, labelCol="indexedLabel")
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, lsvc])
    model = pipeline.fit(df_train)


main()
