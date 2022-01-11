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
from lale.lib.rasl import Aggregate, Map
from lale.lib.sklearn import min_max_scaler
from lale.schemas import Enum


def _df_count(X):
    if _is_pandas_df(X):
        return len(X)
    elif _is_spark_df(X):
        return X.count()


class _MinMaxScalerImpl:
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        if not copy:
            raise ValueError("`copy=False` is not supported by this implementation")
        if clip:
            raise ValueError("`clip=True` is not supported by this implementation")
        self._hyperparams = {"feature_range": feature_range, "copy": copy, "clip": clip}
        self.n_samples_seen_ = 0
        self._transformer = None

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, self._hyperparams))
        return self

    def partial_fit(self, X, y=None):
        if self.n_samples_seen_ == 0:  # first fit
            return self.fit(X)
        lifted_a = (
            self.data_min_,
            self.data_max_,
            self.n_samples_seen_,
            self.n_features_in_,
            self.feature_names_in_,
        )
        lifted_b = self._lift(X, self._hyperparams)
        self._set_fit_attributes(self._combine(lifted_a, lifted_b))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer(X)
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        (
            self.data_min_,
            self.data_max_,
            self.n_samples_seen_,
            self.n_features_in_,
            self.feature_names_in_,
        ) = lifted
        self.data_range_ = self.data_max_ - self.data_min_
        range_min, range_max = self._hyperparams["feature_range"]
        self.scale_ = (range_max - range_min) / (self.data_max_ - self.data_min_)
        self.min_ = range_min - self.data_min_ * self.scale_

    def _build_transformer(self, X):
        range_min, range_max = self._hyperparams["feature_range"]
        ops = {}
        for i, c in enumerate(X.columns):
            c_std = (it[c] - self.data_min_[i]) / (  # type: ignore
                self.data_max_[i] - self.data_min_[i]  # type: ignore
            )
            c_scaled = c_std * (range_max - range_min) + range_min
            ops.update({c: c_scaled})
        return Map(columns=ops)

    @staticmethod
    def _lift(X, hyperparams):
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
        n_samples_seen_ = _df_count(X)
        n_features_in_ = len(X.columns)
        feature_names_in_ = X.columns
        return data_min_, data_max_, n_samples_seen_, n_features_in_, feature_names_in_

    @staticmethod
    def _combine(lifted_a, lifted_b):
        (
            data_min_a,
            data_max_a,
            n_samples_seen_a,
            n_features_in_a,
            feature_names_in_a,
        ) = lifted_a
        (
            data_min_b,
            data_max_b,
            n_samples_seen_b,
            n_features_in_b,
            feature_names_in_b,
        ) = lifted_b
        data_min_ = np.minimum(data_min_a, data_min_b)
        data_max_ = np.maximum(data_max_a, data_max_b)
        n_samples_seen_ = n_samples_seen_a + n_samples_seen_b
        assert n_features_in_a == n_features_in_b
        n_features_in_ = n_features_in_a
        assert list(feature_names_in_a) == list(feature_names_in_b)
        feature_names_in_ = feature_names_in_a
        return data_min_, data_max_, n_samples_seen_, n_features_in_, feature_names_in_


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
