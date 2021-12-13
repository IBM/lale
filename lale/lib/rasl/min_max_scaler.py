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

import numpy as np

import lale.docstrings
import lale.operators
from lale.expressions import it, max, min
from lale.helpers import _is_spark_df
from lale.lib.lale.aggregate import Aggregate
from lale.lib.rasl import Map
from lale.lib.sklearn import min_max_scaler


class _MinMaxScalerImpl:
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        agg = {f"{c}_min": min(it[c]) for c in X.columns}
        agg.update({f"{c}_max": max(it[c]) for c in X.columns})
        aggregate = Aggregate(columns=agg)
        data_min_max = aggregate.transform(X)
        if _is_spark_df(X):
            data_min_max = data_min_max.toPandas()
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = X.columns
        data_min_ = np.zeros(shape=(self.n_features_in_))
        data_max_ = np.zeros(shape=(self.n_features_in_))
        for i, c in enumerate(X.columns):
            data_min_[i] = data_min_max[f"{c}_min"]
            data_max_[i] = data_min_max[f"{c}_max"]
        self.data_min_ = np.array(data_min_)
        self.data_max_ = np.array(data_max_)
        self.data_range_ = self.data_max_ - self.data_min_
        range_min, range_max = self.feature_range
        self.scale_ = (range_max - range_min) / (data_max_ - data_min_)
        self.min_ = range_min - data_min_ * self.scale_
        return self

    def transform(self, X):
        range_min, range_max = self.feature_range
        ops = {}
        for i, c in enumerate(X.columns):
            c_std = (it[c] - self.data_min_[i]) / (  # type: ignore
                self.data_max_[i] - self.data_min_[i]  # type: ignore
            )
            c_scaled = c_std * (range_max - range_min) + range_min
            ops.update({c: c_scaled})
        transformer = Map(columns=ops).fit(X)
        X_transformed = transformer.transform(X)
        return X_transformed


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

lale.docstrings.set_docstrings(MinMaxScaler)
