from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline


def predictions(label_indexer, feature_indexer, train_data, test_data, classifier):
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, classifier])
    model = pipeline.fit(train_data)
    return model.transform(test_data)


def errors(evaluator, prediction, name):
    accuracy = evaluator.evaluate(prediction)
    return f"Test Error {name} = %g " % (1.0 - accuracy)


def specificity_and_sensitivity(prediction):
    prediction_and_labels = prediction.select("indexedLabel", "prediction")
    tp = prediction_and_labels.filter("indexedLabel == '1.0' AND prediction == '1.0'").count()
    fn = prediction_and_labels.filter("indexedLabel == '1.0' AND prediction == '0.0'").count()
    tn = prediction_and_labels.filter("indexedLabel == '0.0' AND prediction == '0.0'").count()
    fp = prediction_and_labels.filter("indexedLabel == '0.0' AND prediction == '1.0'").count()
    return tp / (tp + fn), tn / (tn + fp)


def main():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    feature_assembler = VectorAssembler(inputCols=df.columns[1:], outputCol="features")
    feature_data = feature_assembler.transform(df)

    df_train, df_test = feature_data.randomSplit([0.7, 0.3])
    print("Abnormal Train:", df_train.filter('Status == "Abnormal"').count(), "Normal Train:",
          df_train.filter('Status == "Normal"').count())
    print("Abnormal Test:", df_test.filter('Status == "Abnormal"').count(), "Abnormal Test:",
          df_test.filter('Status == "Normal"').count())
    label_indexer = StringIndexer(inputCol=df.columns[0], outputCol="indexedLabel").fit(feature_data)
    feature_indexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(feature_data)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=[12, 5, 4, 2], labelCol="indexedLabel")
    lsvc = LinearSVC(maxIter=1000, labelCol="indexedLabel")

    prediction_dt = predictions(label_indexer, feature_indexer, df_train, df_test, dt)
    prediction_trainer = predictions(label_indexer, feature_indexer, df_train, df_test, trainer)
    prediction_lsvc = predictions(label_indexer, feature_indexer, df_train, df_test, lsvc)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    print(errors(evaluator, prediction_dt, "Decision Tree"), "sensitivity: ",
          specificity_and_sensitivity(prediction_dt)[0], "specificity: ",
          specificity_and_sensitivity(prediction_dt)[1])
    print(errors(evaluator, prediction_trainer, "ANN"), "sensitivity: ",
          specificity_and_sensitivity(prediction_trainer)[0], "specificity: ",
          specificity_and_sensitivity(prediction_trainer)[1])
    print(errors(evaluator, prediction_lsvc, "Support Vector Machine"), "sensitivity: ",
          specificity_and_sensitivity(prediction_lsvc)[0], "specificity: ",
          specificity_and_sensitivity(prediction_lsvc)[1])


main()
