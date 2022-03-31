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

import lale.docstrings
import lale.helpers
import lale.operators
from lale.expressions import collect_set, it, replace
from lale.helpers import _ensure_pandas
from lale.lib.dataframe import count, get_columns
from lale.lib.sklearn import one_hot_encoder

from .aggregate import Aggregate
from .map import Map
from .monoid import Monoid, MonoidableOperator


class _OneHotEncoderMonoid(Monoid):
    def __init__(self, *, n_samples_seen_, feature_names_in_, categories_):
        self.n_samples_seen_ = n_samples_seen_
        self.feature_names_in_ = feature_names_in_
        self.categories_ = categories_

    def combine(self, other: "_OneHotEncoderMonoid"):
        n_samples_seen_ = self.n_samples_seen_ + other.n_samples_seen_
        assert list(self.feature_names_in_) == list(other.feature_names_in_)
        assert len(self.categories_) == len(other.categories_)
        combined_categories = [
            np.sort(
                np.unique(np.concatenate([self.categories_[i], other.categories_[i]]))
            )
            for i in range(len(self.categories_))
        ]
        return _OneHotEncoderMonoid(
            n_samples_seen_=n_samples_seen_,
            feature_names_in_=self.feature_names_in_,
            categories_=combined_categories,
        )


class _OneHotEncoderImpl(MonoidableOperator[_OneHotEncoderMonoid]):
    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse=False,
        dtype="float64",
        handle_unknown="ignore",
    ):
        self._hyperparams = {
            "categories": categories,
            "drop": drop,
            "sparse": sparse,
            "dtype": dtype,
            "handle_unknown": handle_unknown,
        }

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    @property
    def n_samples_seen_(self):
        return getattr(self._monoid, "n_samples_seen_", 0)

    @property
    def categories_(self):
        return getattr(self._monoid, "categories_", None)

    @property
    def feature_names_in_(self):
        return getattr(self._monoid, "feature_names_in_", None)

    def from_monoid(self, lifted: _OneHotEncoderMonoid):
        self._monoid = lifted
        self.n_features_in_ = len(lifted.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        assert self._monoid is not None
        result = Map(
            columns={
                f"{col_name}_{cat_value}": replace(
                    it[col_name],
                    {cat_value: 1},
                    handle_unknown="use_encoded_value",
                    unknown_value=0,
                )
                for col_idx, col_name in enumerate(self._monoid.feature_names_in_)
                for cat_value in self._monoid.categories_[col_idx]
            }
        )
        return result

    def to_monoid(self, v: Tuple[Any, Any]):
        X, _ = v
        n_samples_seen_ = count(X)
        feature_names_in_ = get_columns(X)
        agg_op = Aggregate(columns={c: collect_set(it[c]) for c in feature_names_in_})
        agg_data = agg_op.transform(X)
        agg_data = _ensure_pandas(agg_data)
        categories_ = [np.sort(agg_data.loc[0, c]) for c in feature_names_in_]
        return _OneHotEncoderMonoid(
            n_samples_seen_=n_samples_seen_,
            feature_names_in_=feature_names_in_,
            categories_=categories_,
        )


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Relational algebra reimplementation of scikit-learn's `OneHotEncoder`_ transformer that encodes categorical features as numbers.
Works on both pandas and Spark dataframes by using `Aggregate`_ for `fit` and `Map`_ for `transform`, which in turn use the appropriate backend.

.. _`OneHotEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
.. _`Aggregate`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.aggregate.html
.. _`Map`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.one_hot_encoder.html",
    "type": "object",
    "tags": {
        "pre": ["categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": one_hot_encoder._hyperparams_schema,
        "input_fit": one_hot_encoder._input_fit_schema,
        "input_transform": one_hot_encoder._input_transform_schema,
        "output_transform": one_hot_encoder._output_transform_schema,
    },
}

OneHotEncoder = lale.operators.make_operator(_OneHotEncoderImpl, _combined_schemas)

OneHotEncoder = typing.cast(
    lale.operators.PlannedIndividualOp,
    OneHotEncoder.customize_schema(
        drop={
            "enum": [None],
            "description": "This implementation only supports `drop=None`.",
            "default": None,
        },
        sparse={
            "enum": [False],
            "description": "This implementation only supports `sparse=False`.",
            "default": False,
        },
        dtype={
            "enum": ["float64"],
            "description": "This implementation only supports `dtype='float64'`.",
            "default": "float64",
        },
        handle_unknown={
            "enum": ["ignore"],
            "description": "This implementation only supports `handle_unknown='ignore'`.",
            "default": "ignore",
        },
    ),
)

lale.docstrings.set_docstrings(OneHotEncoder)
