from collections import Counter
from operator import itemgetter

from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    HasPredictionCol,
)
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import Row


class HasDataset:

    dataset = Param(
        Params._dummy(),
        "dataset",
        "the dataset for calculating the distance",
    )

    def getDataset(self) -> DataFrame:
        return self.getOrDefault(self.dataset)


class HasK(Params):

    k = Param(
        Params._dummy(),
        "k",
        "neightbours",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self):
        super().__init__()
        self._setDefault(k=1)

    def getK(self):
        return self.getOrDefault(self.k)


class KNNClassifier(
    Estimator,
    HasFeaturesCol,
    HasPredictionCol,
    HasLabelCol,
    HasK,
):
    def __init__(self, **kwargs):
        super().__init__()
        self._set(**kwargs)
        self._setDefault(labelCol="label")
        self._setDefault(featuresCol="features")
        self._setDefault(predictionCol="prediction")
        self._setDefault(k=1)

    def _fit(self, dataset):
        return KNNModel(
            dataset=dataset,
            k=self.getK(),
            featuresCol=self.getFeaturesCol(),
            labelCol=self.getLabelCol(),
            predictionCol=self.getPredictionCol(),
        )


class KNNModel(
    Model,
    HasDataset,
    HasFeaturesCol,
    HasPredictionCol,
    HasLabelCol,
    HasK,
):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._set(**kwargs)

    def _transform(self, dataset: DataFrame):

        featuresCol = self.getFeaturesCol()
        predictionCol = self.getPredictionCol()
        k = self.getK()
        labelCol = self.getLabelCol()
        train_set = self.getDataset().select(featuresCol, labelCol).collect()

        def rddfunc(row: Row):
            vector = row[featuresCol]
            data = sorted(
                map(
                    lambda x: (
                        vector.squared_distance(x[featuresCol]),
                        x[labelCol],
                    ),
                    train_set,
                ),
                key=itemgetter(0),
            )
            row_dict = row.asDict()
            row_dict[predictionCol] = Counter(
                map(itemgetter(1), data[:k])
            ).most_common(1)[0][0]
            return Row(**row_dict)

        return dataset.rdd.map(lambda row: rddfunc(row)).toDF()
