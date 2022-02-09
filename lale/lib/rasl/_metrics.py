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

import collections
import functools
from typing import Any, Dict

import numpy as np
import pandas as pd

from lale.datasets.data_schemas import add_table_name
from lale.expressions import astype, count, it, sum
from lale.helpers import _ensure_pandas
from lale.lib.dataframe import get_columns

from .aggregate import Aggregate
from .map import Map

_LiftedAccuracy = collections.namedtuple("_LiftedAccuracy", ["match", "total"])


class _Accuracy:
    def __init__(self):
        from lale.lib.lale import ConcatFeatures

        self._pipeline_suffix = (
            ConcatFeatures
            >> Map(columns={"match": astype("int", it["y_true"] == it["y_pred"])})  # type: ignore
            >> Aggregate(columns={"match": sum(it.match), "total": count(it.match)})
        )

    def _lift(self, batch):
        from lale.lib.lale import Scan

        y_true, y_pred = batch
        assert isinstance(y_true, pd.Series), type(
            y_true
        )  # TODO: make it work on Spark
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, y_true.index, y_true.dtype, "y_pred")
        assert isinstance(y_pred, pd.Series), type(
            y_pred
        )  # TODO: make it work on Spark
        y_true = add_table_name(pd.DataFrame(y_true), "y_true")
        y_pred = add_table_name(pd.DataFrame(y_pred), "y_pred")
        prefix_true = Scan(table=it.y_true) >> Map(
            columns={"y_true": it[get_columns(y_true)[0]]}
        )
        prefix_pred = Scan(table=it.y_pred) >> Map(
            columns={"y_pred": it[get_columns(y_pred)[0]]}
        )
        pipeline = (prefix_true & prefix_pred) >> self._pipeline_suffix
        agg_df = _ensure_pandas(pipeline.transform([y_true, y_pred]))
        return _LiftedAccuracy(agg_df.at[0, "match"], agg_df.at[0, "total"])

    def _combine(self, lifted_a, lifted_b):
        return _LiftedAccuracy(*(a + b for a, b in zip(lifted_a, lifted_b)))

    def _lower(self, lifted):
        return lifted.match / np.float64(lifted.total)

    def score_data(self, y_true, y_pred):
        return self._lower(self._lift((y_true, y_pred)))

    def score_estimator(self, estimator, X, y):
        return self.score_data(y_true=y, y_pred=estimator.predict(X))

    def __call__(self, estimator, X, y):
        return self.score_estimator(estimator, X, y)

    def score_data_batched(self, batches):
        lifted_batches = (self._lift(b) for b in batches)
        return self._lower(functools.reduce(self._combine, lifted_batches))

    def score_estimator_batched(self, estimator, batches):
        predicted_batches = ((y, estimator.predict(X)) for X, y in batches)
        return self.score_data_batched(predicted_batches)


def accuracy_score(y_true, y_pred):
    return get_scorer("accuracy").score_data(y_true, y_pred)


_scorer_cache: Dict[str, Any] = {"accuracy": None}


def get_scorer(scoring: str):
    assert scoring in _scorer_cache, scoring
    if _scorer_cache[scoring] is None:
        if scoring == "accuracy":
            _scorer_cache[scoring] = _Accuracy()
    return _scorer_cache[scoring]
