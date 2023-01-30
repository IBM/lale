# Copyright 2023 IBM Corporation
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
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import scipy.special

import lale.docstrings
import lale.helpers
import lale.operators
from lale.expressions import count as agg_count
from lale.expressions import it, replace
from lale.expressions import sum as agg_sum
from lale.lib.category_encoders import target_encoder
from lale.lib.dataframe import get_columns

from ._util import get_obj_cols
from .aggregate import Aggregate
from .concat_features import ConcatFeatures
from .group_by import GroupBy
from .map import Map
from .monoid import Monoid, MonoidableOperator


class _TargetEncoderMonoid(Monoid):
    def __init__(
        self,
        *,
        feature_names: List[str],
        col2cat2sum: Dict[str, Dict[Union[float, str], float]],
        col2cat2count: Dict[str, Dict[Union[float, str], int]],
    ):
        self.feature_names = feature_names
        assert set(col2cat2sum.keys()) == set(col2cat2count.keys())
        self.col2cat2sum = col2cat2sum
        self.col2cat2count = col2cat2count

    def combine(self, other: "_TargetEncoderMonoid"):
        assert list(self.feature_names) == list(other.feature_names)
        assert set(self.col2cat2sum.keys()) == set(other.col2cat2sum.keys())
        assert set(self.col2cat2count.keys()) == set(other.col2cat2count.keys())
        return _TargetEncoderMonoid(
            feature_names=self.feature_names,
            col2cat2sum={
                col: {
                    cat: cat2sum.get(cat, 0) + other.col2cat2sum[col].get(cat, 0)
                    for cat in set(cat2sum.keys()) | set(other.col2cat2sum[col].keys())
                }
                for col, cat2sum in self.col2cat2sum.items()
            },
            col2cat2count={
                col: {
                    cat: cat2count.get(cat, 0) + other.col2cat2count[col].get(cat, 0)
                    for cat in set(cat2count.keys())
                    | set(other.col2cat2count[col].keys())
                }
                for col, cat2count in self.col2cat2count.items()
            },
        )


class _TargetEncoderImpl(MonoidableOperator[_TargetEncoderMonoid]):
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    @property
    def feature_names(self):
        return self._feature_names

    def from_monoid(self, monoid: _TargetEncoderMonoid):
        self._transformer = None
        self._feature_names = monoid.feature_names
        total_sum = sum(sum(d.values()) for d in monoid.col2cat2sum.values())
        total_count = sum(sum(d.values()) for d in monoid.col2cat2count.values())
        self._prior = total_sum / total_count
        k = self._hyperparams["min_samples_leaf"]
        f = self._hyperparams["smoothing"]

        def blend(posterior, sample_size):
            weighting = scipy.special.expit((sample_size - k) / f)
            return weighting * posterior + (1 - weighting) * self._prior

        self._col2cat2value = {
            col: {
                cat: blend(
                    monoid.col2cat2sum[col][cat] / monoid.col2cat2count[col][cat],
                    sample_size=monoid.col2cat2count[col][cat],
                )
                for cat in monoid.col2cat2sum[col].keys()
            }
            for col in monoid.col2cat2sum.keys()
        }

    def _build_transformer(self):
        categorical_features = self._hyperparams["cols"]

        def build_map_expr(col):
            if col not in categorical_features:
                return it[col]
            return replace(
                it[col],
                self._col2cat2value[col],
                handle_unknown="use_encoded_value",
                unknown_value=self._prior,
            )

        return Map(columns={col: build_map_expr(col) for col in self._feature_names})

    def to_monoid(self, batch: Tuple[Any, Any]):
        X, y = batch
        X_columns = typing.cast(List[str], get_columns(X))
        y_name = lale.helpers.GenSym(set(X_columns))("target")
        if isinstance(y, pd.Series):
            y = pd.DataFrame({y_name: y})
        else:
            y = Map(columns={y_name: it[get_columns(y)[0]]}).transform(y)
        assert lale.helpers._is_df(y)
        classes = self._hyperparams["classes"]
        if classes is not None:
            ordinal_encoder = Map(
                columns={
                    y_name: replace(it[y_name], {v: i for i, v in enumerate(classes)})
                }
            )
            y = ordinal_encoder.transform(y)
        Xy = ConcatFeatures.transform([X, y])
        if self._hyperparams["cols"] is None:
            self._hyperparams["cols"] = get_obj_cols(X)
        col2cat2sum: Dict[str, Dict[Union[float, str], float]] = {}
        col2cat2count: Dict[str, Dict[Union[float, str], int]] = {}
        for col in typing.cast(List[str], self._hyperparams["cols"]):
            pipeline = (
                Map(columns={col: it[col], y_name: it[y_name]})
                >> GroupBy(by=[it[col]])
                >> Aggregate(
                    columns={"sum": agg_sum(it[y_name]), "count": agg_count(it[y_name])}
                )
            )
            aggregated = lale.helpers._ensure_pandas(pipeline.transform(Xy))
            col2cat2sum[col] = aggregated["sum"].to_dict()
            col2cat2count[col] = aggregated["count"].to_dict()
        return _TargetEncoderMonoid(
            feature_names=X_columns,
            col2cat2sum=col2cat2sum,
            col2cat2count=col2cat2count,
        )


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Relational algebra reimplementation of scikit-learn contrib's `TargetEncoder`_ transformer.
Works on both pandas and Spark dataframes by using `Aggregate`_ for `fit` and `Map`_ for `transform`, which in turn use the appropriate backend.

.. _`TargetEncoder`: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
.. _`Aggregate`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.aggregate.html
.. _`Map`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.target_encoder.html",
    "type": "object",
    "tags": {
        "pre": ["categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": target_encoder._hyperparams_schema,
        "input_fit": target_encoder._input_fit_schema,
        "input_transform": target_encoder._input_transform_schema,
        "output_transform": target_encoder._output_transform_schema,
    },
}

TargetEncoder = lale.operators.make_operator(_TargetEncoderImpl, _combined_schemas)

TargetEncoder = typing.cast(
    lale.operators.PlannedIndividualOp,
    TargetEncoder.customize_schema(
        classes={  # TODO: implement classification with >2 classes
            "anyOf": [
                {"enum": [None], "description": "Regression task."},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Binary classification task with numeric labels.",
                    "minItems": 2,
                    "maxItems": 2,
                },
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Binary classification task with string labels.",
                    "minItems": 2,
                    "maxItems": 2,
                },
                {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "description": "Binary classification task with Boolean labels.",
                    "minItems": 2,
                    "maxItems": 2,
                },
            ],
            "default": None,
        },
        drop_invariant={
            "enum": [False],
            "default": False,
            "description": "This implementation only supports `drop_invariant=False`.",
        },
        return_df={
            "enum": [True],
            "default": True,
            "description": "This implementation returns a pandas or spark dataframe if the input is a pandas or spark dataframe, respectively.",
        },
        handle_missing={
            "enum": ["value"],
            "default": "value",
            "description": "This implementation only supports `handle_missing='value'`.",
        },
        handle_unknown={
            "enum": ["value"],
            "default": "value",
            "description": "This implementation only supports `handle_unknown='value'`.",
        },
    ),
)

lale.docstrings.set_docstrings(TargetEncoder)
