# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# %%
df = spark.read.csv("data.csv", inferSchema=True, header=True)


from data import cols, labels

# %%
from pyspark.ml.feature import MaxAbsScaler, VectorAssembler

assembler1 = VectorAssembler(inputCols=cols, outputCol="features")
assembler2 = VectorAssembler(inputCols=labels, outputCol="label")
scaler = MaxAbsScaler(inputCol="features", outputCol="featuresScaled")


# %%
from pyspark.ml.pipeline import Pipeline

pipeline = Pipeline(stages=[assembler1, assembler2, scaler])


# %%
model = pipeline.fit(df)
data = model.transform(df)


# %%
train_set, test_set = data.randomSplit([0.8, 0.2], 1234)


# %%
from knn import KNNClassifier
from pyspark.sql import DataFrame

classifier = KNNClassifier(
    featuresCol="featuresScaled",
    labelCol="label",
    k=1,
    predictionCol="predicted",
)
knn_model = classifier.fit(train_set)
transformed: DataFrame = knn_model.transform(test_set)


# %%
from evaluator import ArrayClassificationEvaluator

eval = ArrayClassificationEvaluator(
    labelCol="label", predictionCol="predicted"
)


# %%
eval.evaluate(transformed)
