# Copyright 2021 IBM Corporation
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

import typing

import numpy as np

import lale.docstrings
import lale.operators
from lale.expressions import it
from lale.expressions import max as agg_max
from lale.expressions import min as agg_min
from lale.helpers import _is_pandas_df, _is_spark_df
from lale.lib.lale.aggregate import Aggregate
from lale.lib.rasl import Map
from lale.lib.sklearn import min_max_scaler
from lale.schemas import Enum


def _df_count(X):
    if _is_pandas_df(X):
        return len(X)
    elif _is_spark_df(X):
        return X.count()


class _MinMaxScalerImpl:
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        if not copy:
            raise ValueError("`copy=False` is not supported by this implementation")
        if clip:
            raise ValueError("`clip=True` is not supported by this implementation")
        self._first_fit = True

    def fit(self, X, y=None):
        data_min_, data_max_ = self._get_min_max(X)
        n_samples_seen_ = _df_count(X)
        self._set_fit_attributes(X, data_min_, data_max_, n_samples_seen_)
        self.transformer = self._build_transformer(X)

    def partial_fit(self, X, y=None):
        data_min_, data_max_ = self._get_min_max(X)
        if self._first_fit:
            n_samples_seen_ = _df_count(X)
        else:
            data_min_ = np.minimum(data_min_, self.data_min_)
            data_max_ = np.maximum(data_max_, self.data_max_)
            n_samples_seen_ = _df_count(X) + self.n_samples_seen_
        self._set_fit_attributes(X, data_min_, data_max_, n_samples_seen_)
        self.transformer = self._build_transformer(X)

    def _get_min_max(self, X):
        agg = {f"{c}_min": agg_min(it[c]) for c in X.columns}
        agg.update({f"{c}_max": agg_max(it[c]) for c in X.columns})
        aggregate = Aggregate(columns=agg)
        data_min_max = aggregate.transform(X)
        if _is_spark_df(X):
            data_min_max = data_min_max.toPandas()
        n = len(X.columns)
        data_min_ = np.zeros(shape=(n))
        data_max_ = np.zeros(shape=(n))
        for i, c in enumerate(X.columns):
            data_min_[i] = data_min_max[f"{c}_min"]
            data_max_[i] = data_min_max[f"{c}_max"]
        data_min_ = np.array(data_min_)
        data_max_ = np.array(data_max_)
        return data_min_, data_max_

    def _set_fit_attributes(self, X, data_min_, data_max_, n_samples_seen_):
        self._first_fit = False
        self.data_min_ = data_min_
        self.data_max_ = data_max_
        self.n_samples_seen_ = n_samples_seen_
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = X.columns
        self.data_range_ = self.data_max_ - self.data_min_
        range_min, range_max = self.feature_range
        self.scale_ = (range_max - range_min) / (data_max_ - data_min_)
        self.min_ = range_min - data_min_ * self.scale_

    def _build_transformer(self, X):
        range_min, range_max = self.feature_range
        ops = {}
        for i, c in enumerate(X.columns):
            c_std = (it[c] - self.data_min_[i]) / (  # type: ignore
                self.data_max_[i] - self.data_min_[i]  # type: ignore
            )
            c_scaled = c_std * (range_max - range_min) + range_min
            ops.update({c: c_scaled})
        return Map(columns=ops).fit(X)

    def transform(self, X):
        return self.transformer.transform(X)


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra implementation of MinMaxScaler.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.min_max_scaler.html",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": min_max_scaler._hyperparams_schema,
        "input_fit": min_max_scaler._input_schema_fit,
        "input_transform": min_max_scaler._input_transform_schema,
        "output_transform": min_max_scaler._output_transform_schema,
    },
}

MinMaxScaler = lale.operators.make_operator(_MinMaxScalerImpl, _combined_schemas)

MinMaxScaler = typing.cast(
    lale.operators.PlannedIndividualOp,
    MinMaxScaler.customize_schema(
        copy=Enum(
            values=[True],
            desc="`copy=True` is the only value currently supported by this implementation",
            default=True,
        ),
        clip=Enum(
            values=[False],
            desc="`clip=False` is the only value currently supported by this implementation",
            default=False,
        ),
    ),
)

lale.docstrings.set_docstrings(MinMaxScaler)
