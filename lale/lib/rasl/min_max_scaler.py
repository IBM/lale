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
from typing import Any, Tuple

import numpy as np

try:
    from sklearn.preprocessing._data import _handle_zeros_in_scale
except ModuleNotFoundError:

    def _handle_zeros_in_scale(scale):
        constant_mask = scale < 10 * np.finfo(scale.dtype).eps
        scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale


import lale.docstrings
import lale.operators
from lale.datasets.data_schemas import forward_metadata
from lale.expressions import it
from lale.expressions import max as agg_max
from lale.expressions import min as agg_min
from lale.helpers import _is_spark_df
from lale.lib.dataframe import count, get_columns
from lale.lib.rasl import Aggregate, Map
from lale.lib.sklearn import min_max_scaler
from lale.schemas import Enum

from .monoid import Monoid, MonoidableOperator


class _MinMaxScalerMonoid(Monoid):
    def __init__(
        self,
        *,
        data_min_: np.ndarray,
        data_max_: np.ndarray,
        n_samples_seen_: int,
        feature_names_in_,
    ):
        self.data_min_ = data_min_
        self.data_max_ = data_max_
        self.n_samples_seen_ = n_samples_seen_
        self.feature_names_in_ = feature_names_in_

    def combine(self, other: "_MinMaxScalerMonoid"):
        data_min_ = np.minimum(self.data_min_, other.data_min_)
        data_max_ = np.maximum(self.data_max_, other.data_max_)
        n_samples_seen_ = self.n_samples_seen_ + other.n_samples_seen_
        assert list(self.feature_names_in_) == list(self.feature_names_in_)
        feature_names_in_ = self.feature_names_in_
        return _MinMaxScalerMonoid(
            data_min_=data_min_,
            data_max_=data_max_,
            n_samples_seen_=n_samples_seen_,
            feature_names_in_=feature_names_in_,
        )


class _MinMaxScalerImpl(MonoidableOperator[_MinMaxScalerMonoid]):
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        if not copy:
            raise ValueError("`copy=False` is not supported by this implementation")
        if clip:
            raise ValueError("`clip=True` is not supported by this implementation")
        self._hyperparams = {"feature_range": feature_range, "copy": copy, "clip": clip}

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        X_new = self._transformer.transform(X)
        return forward_metadata(X, X_new)

    @property
    def data_min_(self):
        return getattr(self._monoid, "data_min_", None)

    @property
    def data_max_(self):
        return getattr(self._monoid, "data_max_", None)

    @property
    def n_samples_seen_(self):
        return getattr(self._monoid, "n_samples_seen_", 0)

    @property
    def feature_names_in_(self):
        return getattr(self._monoid, "feature_names_in_", None)

    def from_monoid(self, v: _MinMaxScalerMonoid):
        self._monoid = v
        self.n_features_in_ = len(v.feature_names_in_)
        self.data_range_ = v.data_max_ - v.data_min_  # type: ignore
        range_min, range_max = self._hyperparams["feature_range"]
        self.scale_ = (range_max - range_min) / _handle_zeros_in_scale(v.data_max_ - v.data_min_)  # type: ignore
        self.min_ = range_min - v.data_min_ * self.scale_
        self._transformer = None

    def _build_transformer(self):
        range_min, range_max = self._hyperparams["feature_range"]
        ops = {}
        dmin = self.data_min_
        assert dmin is not None
        dmax = self.data_max_
        assert dmax is not None
        for i, c in enumerate(self.feature_names_in_):
            c_std = (it[c] - dmin[i]) / _handle_zeros_in_scale(dmax[i] - dmin[i])
            c_scaled = c_std * (range_max - range_min) + range_min
            ops.update({c: c_scaled})
        return Map(columns=ops)

    def to_monoid(self, v: Tuple[Any, Any]) -> _MinMaxScalerMonoid:
        X, _ = v
        X_cols = get_columns(X)
        agg = {f"{c}_min": agg_min(it[c]) for c in X_cols}
        agg.update({f"{c}_max": agg_max(it[c]) for c in X_cols})
        aggregate = Aggregate(columns=agg)
        data_min_max = aggregate.transform(X)
        if _is_spark_df(X):
            data_min_max = data_min_max.toPandas()
        n = len(X_cols)
        data_min_ = np.zeros(shape=(n))
        data_max_ = np.zeros(shape=(n))
        for i, c in enumerate(X_cols):
            data_min_[i] = data_min_max[f"{c}_min"]
            data_max_[i] = data_min_max[f"{c}_max"]
        data_min_ = np.array(data_min_)
        data_max_ = np.array(data_max_)
        n_samples_seen_ = count(X)
        feature_names_in_ = X_cols
        return _MinMaxScalerMonoid(
            data_min_=data_min_,
            data_max_=data_max_,
            n_samples_seen_=n_samples_seen_,
            feature_names_in_=feature_names_in_,
        )


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
