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
from lale.lib.dataframe import get_columns
from lale.lib.sklearn import standard_scaler

from .aggregate import Aggregate
from .map import Map
from .monoid import Monoid, MonoidableOperator


class _StandardScalerMonoid(Monoid):
    def __init__(self, *, feature_names_in_, n_samples_seen_, _sum1, _sum2):
        self.feature_names_in_ = feature_names_in_
        self.n_samples_seen_ = n_samples_seen_
        self._sum1 = _sum1
        self._sum2 = _sum2

    def combine(self, other):
        assert list(self.feature_names_in_) == list(other.feature_names_in_)
        combined_feat = self.feature_names_in_
        combined_n_samples_seen = self.n_samples_seen_ + other.n_samples_seen_
        if self._sum1 is None:
            combined_sum1 = None
        else:
            assert other._sum1 is not None and len(self._sum1) == len(other._sum1)
            combined_sum1 = [
                self._sum1[i] + other._sum1[i] for i in range(len(self._sum1))
            ]
        if self._sum2 is None:
            combined_sum2 = None
        else:
            assert other._sum2 is not None and len(self._sum2) == len(other._sum2)
            combined_sum2 = [
                self._sum2[i] + other._sum2[i] for i in range(len(self._sum2))
            ]
        return _StandardScalerMonoid(
            feature_names_in_=combined_feat,
            n_samples_seen_=combined_n_samples_seen,
            _sum1=combined_sum1,
            _sum2=combined_sum2,
        )


class _StandardScalerImpl(MonoidableOperator[_StandardScalerMonoid]):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self._hyperparams = {"copy": copy, "with_mean": with_mean, "with_std": with_std}

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    @property
    def n_samples_seen_(self):
        return getattr(self._monoid, "n_samples_seen_", 0)

    @property
    def feature_names_in_(self):
        return getattr(self._monoid, "feature_names_in_", None)

    def from_monoid(self, lifted):
        self._monoid = lifted
        n = lifted.n_samples_seen_
        if self._hyperparams["with_std"]:
            # Table 1 of http://www.vldb.org/pvldb/vol8/p702-tangwongsan.pdf
            self.var_ = [
                (lifted._sum2[i] - lifted._sum1[i] * lifted._sum1[i] / n) / n
                for i in range(len(lifted._sum1))
            ]
            self.scale_ = [
                1.0 if self.var_[i] == 0.0 else np.sqrt(self.var_[i])
                for i in range(len(lifted._sum1))
            ]
        else:
            self.var_ = None
            self.scale_ = None
        if self._hyperparams["with_mean"]:
            self.mean_ = [lifted._sum1[i] / n for i in range(len(lifted._sum1))]
        else:
            self.mean_ = None
        self.n_features_in_ = len(lifted.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        def scale_expr(col_idx, col_name):
            expr = it[col_name]
            if self.mean_ is not None:
                expr = expr - self.mean_[col_idx]
            if self.scale_ is not None:
                expr = expr / self.scale_[col_idx]
            return expr

        assert self._monoid is not None
        result = Map(
            columns={
                col_name: scale_expr(col_idx, col_name)
                for col_idx, col_name in enumerate(self._monoid.feature_names_in_)
            }
        )
        return result

    def to_monoid(self, v):
        X, _ = v
        hyperparams = self._hyperparams
        feature_names_in = get_columns(X)
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
        return _StandardScalerMonoid(
            feature_names_in_=feature_names_in,
            n_samples_seen_=n_samples_seen,
            _sum1=sum1,
            _sum2=sum2,
        )


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
