# Copyright 2022 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from abc import abstractmethod
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
from typing_extensions import Protocol, TypeAlias

from lale.expressions import astype, count, it
from lale.expressions import sum as lale_sum
from lale.helpers import spark_installed
from lale.operators import TrainedOperator

from .aggregate import Aggregate
from .group_by import GroupBy
from .map import Map
from .monoid import Monoid, MonoidFactory

MetricMonoid = Monoid

_M = TypeVar("_M", bound=MetricMonoid)

_PandasBatch: TypeAlias = Tuple[pd.DataFrame, pd.Series]

if spark_installed:
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame

    _SparkBatch: TypeAlias = Tuple[SparkDataFrame, SparkDataFrame]

    _Batch_XyAux = Union[_PandasBatch, _SparkBatch]

    _Batch_yyXAux = Tuple[
        Union[pd.Series, np.ndarray, SparkDataFrame],
        Union[pd.Series, np.ndarray, SparkDataFrame],
        Union[pd.DataFrame, SparkDataFrame],
    ]

else:
    _Batch_XyAux = _PandasBatch  # type: ignore

    _Batch_yyXAux = Tuple[  # type: ignore
        Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], pd.DataFrame
    ]

# pyright does not currently accept a TypeAlias with conditional definitions
_Batch_Xy: TypeAlias = _Batch_XyAux  # type: ignore
_Batch_yyX: TypeAlias = _Batch_yyXAux  # type: ignore


class MetricMonoidFactory(MonoidFactory[_Batch_yyX, float, _M], Protocol):
    """Abstract base class for factories that create metrics with an associative monoid interface."""

    @abstractmethod
    def to_monoid(self, batch: _Batch_yyX) -> _M:
        pass

    @abstractmethod
    def score_data(
        self, y_true: pd.Series, y_pred: pd.Series, X: Optional[pd.DataFrame] = None
    ) -> float:
        pass  # keeping this abstract to allow inheriting non-batched version

    @abstractmethod
    def score_estimator(
        self, estimator: TrainedOperator, X: pd.DataFrame, y: pd.Series
    ) -> float:
        pass  # keeping this abstract to allow inheriting non-batched version

    def __call__(
        self, estimator: TrainedOperator, X: pd.DataFrame, y: pd.Series
    ) -> float:
        return self.score_estimator(estimator, X, y)

    def score_data_batched(self, batches: Iterable[_Batch_yyX]) -> float:
        lifted_batches = (self.to_monoid(b) for b in batches)
        combined = functools.reduce(lambda a, b: a.combine(b), lifted_batches)
        return self.from_monoid(combined)

    def score_estimator_batched(
        self, estimator: TrainedOperator, batches: Iterable[_Batch_Xy]
    ) -> float:
        predicted_batches = ((y, estimator.predict(X), X) for X, y in batches)
        return self.score_data_batched(predicted_batches)


class _MetricMonoidMixin(MetricMonoidFactory[_M], Protocol):
    # pylint:disable=abstract-method
    # This is an abstract class as well
    def score_data(
        self, y_true: pd.Series, y_pred: pd.Series, X: Optional[pd.DataFrame] = None
    ) -> float:
        return self.from_monoid(self.to_monoid((y_true, y_pred, X)))

    def score_estimator(
        self, estimator: TrainedOperator, X: pd.DataFrame, y: pd.Series
    ) -> float:
        return self.score_data(y_true=y, y_pred=estimator.predict(X), X=X)


def _make_dataframe_yy(batch):
    def make_series_y(y):
        if isinstance(y, np.ndarray):
            series = pd.Series(y)
        elif isinstance(y, pd.DataFrame):
            series = y.squeeze()
        elif spark_installed and isinstance(y, SparkDataFrame):
            series = cast(pd.DataFrame, y.toPandas()).squeeze()
        else:
            series = y
        assert isinstance(series, pd.Series), type(series)
        return series.reset_index(drop=True)

    y_true, y_pred, _ = batch
    result = pd.DataFrame(
        {"y_true": make_series_y(y_true), "y_pred": make_series_y(y_pred)},
    )
    return result


class _AccuracyData(MetricMonoid):
    def __init__(self, match: int, total: int):
        self.match = match
        self.total = total

    def combine(self, other: "_AccuracyData") -> "_AccuracyData":
        return _AccuracyData(self.match + other.match, self.total + other.total)


class _Accuracy(_MetricMonoidMixin[_AccuracyData]):
    def __init__(self):
        self._pipeline = Map(
            columns={"match": astype("int", it.y_true == it.y_pred)}
        ) >> Aggregate(columns={"match": lale_sum(it.match), "total": count(it.match)})

    def to_monoid(self, batch: _Batch_yyX) -> _AccuracyData:
        input_df = _make_dataframe_yy(batch)
        agg_df = self._pipeline.transform(input_df)
        return _AccuracyData(match=agg_df.at[0, "match"], total=agg_df.at[0, "total"])

    def from_monoid(self, monoid: _AccuracyData) -> float:
        return float(monoid.match / np.float64(monoid.total))


def accuracy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Replacement for sklearn's `accuracy_score`_ function.

    .. _`accuracy_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    """
    return get_scorer("accuracy").score_data(y_true, y_pred)


class _BalancedAccuracyData(MetricMonoid):
    def __init__(self, true_pos: Dict[str, int], false_neg: Dict[str, int]):
        self.true_pos = true_pos
        self.false_neg = false_neg

    def combine(self, other: "_BalancedAccuracyData") -> "_BalancedAccuracyData":
        keys = set(self.true_pos.keys()) | set(other.true_pos.keys())
        return _BalancedAccuracyData(
            {k: self.true_pos.get(k, 0) + other.true_pos.get(k, 0) for k in keys},
            {k: self.false_neg.get(k, 0) + other.false_neg.get(k, 0) for k in keys},
        )


class _BalancedAccuracy(_MetricMonoidMixin[_BalancedAccuracyData]):
    def __init__(self):
        self._pipeline = (
            Map(
                columns={
                    "y_true": it.y_true,
                    "true_pos": astype("int", (it.y_pred == it.y_true)),
                    "false_neg": astype("int", (it.y_pred != it.y_true)),
                }
            )
            >> GroupBy(by=[it.y_true])
            >> Aggregate(
                columns={
                    "true_pos": lale_sum(it.true_pos),
                    "false_neg": lale_sum(it.false_neg),
                }
            )
        )

    def to_monoid(self, batch: _Batch_yyX) -> _BalancedAccuracyData:
        input_df = _make_dataframe_yy(batch)
        agg_df = self._pipeline.transform(input_df)
        return _BalancedAccuracyData(
            true_pos={k: agg_df.at[k, "true_pos"] for k in agg_df.index},
            false_neg={k: agg_df.at[k, "false_neg"] for k in agg_df.index},
        )

    def from_monoid(self, monoid: _BalancedAccuracyData) -> float:
        recalls = {
            k: monoid.true_pos[k] / (monoid.true_pos[k] + monoid.false_neg[k])
            for k in monoid.true_pos
        }
        result = sum(recalls.values()) / len(recalls)
        return float(result)


def balanced_accuracy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Replacement for sklearn's `balanced_accuracy_score`_ function.

    .. _`balanced_accuracy_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """
    return get_scorer("balanced_accuracy").score_data(y_true, y_pred)


class _F1Data(MetricMonoid):
    def __init__(self, true_pos: int, false_pos: int, false_neg: int):
        self.true_pos = true_pos
        self.false_pos = false_pos
        self.false_neg = false_neg

    def combine(self, other: "_F1Data") -> "_F1Data":
        return _F1Data(
            self.true_pos + other.true_pos,
            self.false_pos + other.false_pos,
            self.false_neg + other.false_neg,
        )


class _F1(_MetricMonoidMixin[_F1Data]):
    def __init__(self, pos_label: Union[int, float, str] = 1):
        self._pipeline = Map(
            columns={
                "true_pos": astype(
                    "int", (it.y_pred == pos_label) & (it.y_true == pos_label)
                ),
                "false_pos": astype(
                    "int", (it.y_pred == pos_label) & (it.y_true != pos_label)
                ),
                "false_neg": astype(
                    "int", (it.y_pred != pos_label) & (it.y_true == pos_label)
                ),
            }
        ) >> Aggregate(
            columns={
                "true_pos": lale_sum(it.true_pos),
                "false_pos": lale_sum(it.false_pos),
                "false_neg": lale_sum(it.false_neg),
            }
        )

    def to_monoid(self, batch: _Batch_yyX) -> _F1Data:
        input_df = _make_dataframe_yy(batch)
        agg_df = self._pipeline.transform(input_df)
        return _F1Data(
            true_pos=agg_df.at[0, "true_pos"],
            false_pos=agg_df.at[0, "false_pos"],
            false_neg=agg_df.at[0, "false_neg"],
        )

    def from_monoid(self, monoid: _F1Data) -> float:
        two_tp = monoid.true_pos + monoid.true_pos
        result = two_tp / (two_tp + monoid.false_pos + monoid.false_neg)
        return float(result)


def f1_score(
    y_true: pd.Series, y_pred: pd.Series, pos_label: Union[int, float, str] = 1
) -> float:
    """Replacement for sklearn's `f1_score`_ function.

    .. _`f1_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    return get_scorer("f1", pos_label=pos_label).score_data(y_true, y_pred)


class _R2Data(MetricMonoid):
    def __init__(self, n: int, tot_sum: float, tot_sum_sq: float, res_sum_sq: float):
        self.n = n
        self.sum = tot_sum
        self.sum_sq = tot_sum_sq
        self.res_sum_sq = res_sum_sq

    def combine(self, other: "_R2Data") -> "_R2Data":
        return _R2Data(
            n=self.n + other.n,
            tot_sum=self.sum + other.sum,
            tot_sum_sq=self.sum_sq + other.sum_sq,
            res_sum_sq=self.res_sum_sq + other.res_sum_sq,
        )


class _R2(_MetricMonoidMixin[_R2Data]):
    # https://en.wikipedia.org/wiki/Coefficient_of_determination

    def __init__(self):
        self._pipeline = Map(
            columns={
                "y": it.y_true,  # observed values
                "f": it.y_pred,  # predicted values
                "y2": it.y_true * it.y_true,  # squares
                "e2": (it.y_true - it.y_pred) * (it.y_true - it.y_pred),
            }
        ) >> Aggregate(
            columns={
                "n": count(it.y),
                "sum": lale_sum(it.y),
                "sum_sq": lale_sum(it.y2),
                "res_sum_sq": lale_sum(it.e2),  # residual sum of squares
            }
        )

    def to_monoid(self, batch: _Batch_yyX) -> _R2Data:
        input_df = _make_dataframe_yy(batch)
        agg_df = self._pipeline.transform(input_df)
        return _R2Data(
            n=agg_df.at[0, "n"],
            tot_sum=agg_df.at[0, "sum"],
            tot_sum_sq=agg_df.at[0, "sum_sq"],
            res_sum_sq=agg_df.at[0, "res_sum_sq"],
        )

    def from_monoid(self, monoid: _R2Data) -> float:
        ss_tot = monoid.sum_sq - (monoid.sum * monoid.sum / np.float64(monoid.n))
        return 1 - float(monoid.res_sum_sq / ss_tot)


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Replacement for sklearn's `r2_score`_ function.

    .. _`r2_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    """
    return get_scorer("r2").score_data(y_true, y_pred)


_scorer_cache: Dict[str, Optional[MetricMonoidFactory]] = {
    "accuracy": None,
    "balanced_accuracy": None,
    "r2": None,
}


def get_scorer(scoring: str, **kwargs) -> MetricMonoidFactory:
    """Replacement for sklearn's `get_scorer`_ function.

    .. _`get_scorer`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html
    """
    if scoring == "f1":
        return _F1(**kwargs)
    assert scoring in _scorer_cache, scoring
    if _scorer_cache[scoring] is None:
        if scoring == "accuracy":
            _scorer_cache[scoring] = _Accuracy()
        elif scoring == "balanced_accuracy":
            _scorer_cache[scoring] = _BalancedAccuracy()
        elif scoring == "r2":
            _scorer_cache[scoring] = _R2()
    result = _scorer_cache[scoring]
    assert result is not None
    return result
