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
import lale.helpers
import lale.operators
from lale.expressions import count, it
from lale.expressions import sum as agg_sum
from lale.lib.sklearn import standard_scaler

from .aggregate import Aggregate
from .map import Map


class _StandardScalerImpl:
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self._hyperparams = {"copy": copy, "with_mean": with_mean, "with_std": with_std}
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, self._hyperparams))
        return self

    def partial_fit(self, X, y=None):
        if self.n_samples_seen_ == 0:  # first fit
            return self.fit(X)
        lifted_a = self.feature_names_in_, self.n_samples_seen_, self._sum1, self._sum2
        lifted_b = self._lift(X, self._hyperparams)
        self._set_fit_attributes(self._combine(lifted_a, lifted_b))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        self.feature_names_in_, self.n_samples_seen_, self._sum1, self._sum2 = lifted
        n = self.n_samples_seen_
        if self._hyperparams["with_std"]:
            # Table 1 of http://www.vldb.org/pvldb/vol8/p702-tangwongsan.pdf
            self.var_ = [
                (self._sum2[i] - self._sum1[i] * self._sum1[i] / n) / n
                for i in range(len(self._sum1))
            ]
            self.scale_ = [
                1.0 if self.var_[i] == 0.0 else np.sqrt(self.var_[i])
                for i in range(len(self._sum1))
            ]
        else:
            self.var_ = None
            self.scale_ = None
        if self._hyperparams["with_mean"]:
            self.mean_ = [self._sum1[i] / n for i in range(len(self._sum1))]
        else:
            self.mean_ = None
        self.n_features_in_ = len(self.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        def scale_expr(col_idx, col_name):
            expr = it[col_name]
            if self.mean_ is not None:
                expr = expr - self.mean_[col_idx]
            if self.scale_ is not None:
                expr = expr / self.scale_[col_idx]
            return expr

        result = Map(
            columns={
                col_name: scale_expr(col_idx, col_name)
                for col_idx, col_name in enumerate(self.feature_names_in_)
            }
        )
        return result

    @staticmethod
    def _lift(X, hyperparams):
        feature_names_in = X.columns
        count_op = Aggregate(columns={"count": count(it[feature_names_in[0]])})
        count_data = lale.helpers._ensure_pandas(count_op.transform(X))
        n_samples_seen = count_data.loc[0, "count"]
        if hyperparams["with_mean"] or hyperparams["with_std"]:
            sum1_op = Aggregate(columns={c: agg_sum(it[c]) for c in feature_names_in})
            sum1_data = lale.helpers._ensure_pandas(sum1_op.transform(X))
            sum1 = [sum1_data.loc[0, c] for c in feature_names_in]
        else:
            sum1 = None
        if hyperparams["with_std"]:
            sum2_op = Map(
                columns={c: it[c] * it[c] for c in feature_names_in}
            ) >> Aggregate(columns={c: agg_sum(it[c]) for c in feature_names_in})
            sum2_data = lale.helpers._ensure_pandas(sum2_op.transform(X))
            sum2 = [sum2_data.loc[0, c] for c in feature_names_in]
        else:
            sum2 = None
        return feature_names_in, n_samples_seen, sum1, sum2

    @staticmethod
    def _combine(lifted_a, lifted_b):
        feature_names_in_a, n_samples_seen_a, sum1_a, sum2_a = lifted_a
        feature_names_in_b, n_samples_seen_b, sum1_b, sum2_b = lifted_b
        assert list(feature_names_in_a) == list(feature_names_in_b)
        combined_feat = feature_names_in_a
        combined_n_samples_seen = n_samples_seen_a + n_samples_seen_b
        if sum1_a is None:
            combined_sum1 = None
        else:
            assert sum1_b is not None and len(sum1_a) == len(sum1_b)
            combined_sum1 = [sum1_a[i] + sum1_b[i] for i in range(len(sum1_a))]
        if sum2_a is None:
            combined_sum2 = None
        else:
            assert sum2_b is not None and len(sum2_a) == len(sum2_b)
            combined_sum2 = [sum2_a[i] + sum2_b[i] for i in range(len(sum2_a))]
        return combined_feat, combined_n_samples_seen, combined_sum1, combined_sum2


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Relational algebra reimplementation of scikit-learn's `StandardScaler`_ transformer that standardizes features by removing the mean and scaling to unit variance.
Works on both pandas and Spark dataframes by using `Aggregate`_ for `fit` and `Map`_ for `transform`, which in turn use the appropriate backend.

.. _`StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
.. _`Aggregate`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.aggregate.html
.. _`Map`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.standard_scaler.html",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": standard_scaler._hyperparams_schema,
        "input_fit": standard_scaler._input_fit_schema,
        "input_transform": standard_scaler._input_transform_schema,
        "output_transform": standard_scaler._output_transform_schema,
    },
}

StandardScaler = lale.operators.make_operator(_StandardScalerImpl, _combined_schemas)

StandardScaler = typing.cast(
    lale.operators.PlannedIndividualOp,
    StandardScaler.customize_schema(
        copy={
            "enum": [True],
            "description": "This implementation only supports `copy=True`.",
            "default": True,
        },
    ),
)

lale.docstrings.set_docstrings(StandardScaler)
