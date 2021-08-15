from pyspark.ml.evaluation import Evaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol
from pyspark.sql.dataframe import DataFrame


class ArrayClassificationEvaluator(
    Evaluator,
    HasLabelCol,
    HasPredictionCol,
):

    metricName = Param(
        Params._dummy(), "metricName", "accuracy", TypeConverters.toString
    )

    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(metricName="accuracy")
        self._set(**kwargs)

    def _evaluate(self, dataset: DataFrame):
        return {"accuracy": lambda df: self.accuracy(df),}[
            self.metricName
        ](dataset)

    def metric(self, df: DataFrame, fn=None):
        if fn is None:
            fn = self._accuracy
        length = len(df.first()[self.getLabelCol()])
        data = {}
        total = df.count()
        for idx in range(length):
            data[idx] = fn(df, idx, total)

        return data

    def accuracy(self, df: DataFrame):
        return self.metric(df, self._accuracy)

    def _accuracy(self, df: DataFrame, label_idx: int, total: int):
        correct = df.filter(
            vector_to_array(df[self.getLabelCol()])[label_idx]
            == vector_to_array(df[self.getPredictionCol()])[label_idx]
        ).count()
        return correct / total
