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
from lale.expressions import collect_set, it, replace
from lale.lib.sklearn import one_hot_encoder

from .aggregate import Aggregate
from .map import Map


class _OneHotEncoderImpl:
    def __init__(
        self,
        categories="auto",
        drop=None,
        sparse=False,
        dtype="float64",
        handle_unknown="ignore",
    ):
        self.categories = categories
        if categories != "auto":
            self.categories_ = categories
        if drop is not None:
            raise ValueError("This implementation only supports `drop=None`.")
        if sparse:
            raise ValueError("This implementation only supports `sparse=False`.")
        if dtype != "float64":
            raise ValueError("This implementation only supports `dtype='float64'`.")
        if handle_unknown != "ignore":
            raise ValueError(
                "This implementation only supports `handle_unknown='ignore'`."
            )
        self._transformer = None

    def fit(self, X, y=None):
        if self.categories == "auto":
            self.feature_names_in_, self.categories_ = self._lift(X)
            self._transformer = None
        return self

    def partial_fit(self, X, y=None):
        if not hasattr(self, "categories_"):  # first fit
            return self.fit(X)
        if self.categories == "auto":
            lifted1 = self.feature_names_in_, self.categories_
            lifted2 = self._lift(X)
            self.feature_names_in_, self.categories_ = self._combine(lifted1, lifted2)
            self._transformer = None
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._lower((self.feature_names_in_, self.categories_))
        return self._transformer.transform(X)

    @staticmethod
    def _lift(X):
        feature_names_in = X.columns
        agg_op = Aggregate(columns={c: collect_set(it[c]) for c in feature_names_in})
        agg_data = agg_op.transform(X)
        if lale.helpers._is_spark_df(agg_data):
            agg_data = agg_data.toPandas()
        categories = [np.sort(agg_data.loc[0, c]) for c in feature_names_in]
        return feature_names_in, categories

    @staticmethod
    def _combine(lifted1, lifted2):
        feature_names_in1, categories1 = lifted1
        feature_names_in2, categories2 = lifted2
        assert list(feature_names_in1) == list(feature_names_in2)
        assert len(categories1) == len(categories2)
        combined_categories = [
            np.sort(np.unique(np.concatenate([categories1[i], categories2[i]])))
            for i in range(len(categories1))
        ]
        return feature_names_in1, combined_categories

    @staticmethod
    def _lower(lifted):
        feature_names_in, categories = lifted
        result = Map(
            columns={
                f"{col_name}_{cat_value}": replace(
                    it[col_name],
                    {cat_value: 1},
                    handle_unknown="use_encoded_value",
                    unknown_value=0,
                )
                for col_idx, col_name in enumerate(feature_names_in)
                for cat_value in categories[col_idx]
            }
        )
        return result


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
        "pre": ["~categoricals"],
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
