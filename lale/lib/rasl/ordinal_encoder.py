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
from lale.lib.sklearn import ordinal_encoder

from .aggregate import Aggregate
from .map import Map


class _OrdinalEncoderImpl:
    def __init__(
        self,
        *,
        categories="auto",
        dtype="float64",
        handle_unknown="error",
        unknown_value=None,
    ):
        self._hyperparams = {
            "categories": categories,
            "dtype": dtype,
            "handle_unknown": handle_unknown,
            "unknown_value": unknown_value,
        }

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, self._hyperparams))
        return self

    def partial_fit(self, X, y=None):
        if not hasattr(self, "categories_"):  # first fit
            return self.fit(X)
        lifted_a = self.feature_names_in_, self.categories_
        lifted_b = self._lift(X, self._hyperparams)
        self._set_fit_attributes(self._combine(lifted_a, lifted_b))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        self.feature_names_in_, self.categories_ = lifted
        self.n_features_in_ = len(self.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        result = Map(
            columns={
                col_name: replace(
                    it[col_name],
                    {
                        cat_value: cat_idx
                        for cat_idx, cat_value in enumerate(self.categories_[col_idx])
                    },
                    handle_unknown="use_encoded_value",
                    unknown_value=self._hyperparams["unknown_value"],
                )
                for col_idx, col_name in enumerate(self.feature_names_in_)
            }
        )
        return result

    @staticmethod
    def _lift(X, hyperparams):
        feature_names_in = X.columns
        if hyperparams["categories"] == "auto":
            agg_op = Aggregate(
                columns={c: collect_set(it[c]) for c in feature_names_in}
            )
            agg_data = agg_op.transform(X)
            if lale.helpers._is_spark_df(agg_data):
                agg_data = agg_data.toPandas()
            categories = [np.sort(agg_data.loc[0, c]) for c in feature_names_in]
        else:
            categories = hyperparams["categories"]
        return feature_names_in, categories

    @staticmethod
    def _combine(lifted_a, lifted_b):
        feature_names_in_a, categories_a = lifted_a
        feature_names_in_b, categories_b = lifted_b
        assert list(feature_names_in_a) == list(feature_names_in_b)
        assert len(categories_a) == len(categories_b)
        combined_categories = [
            np.sort(np.unique(np.concatenate([categories_a[i], categories_b[i]])))
            for i in range(len(categories_a))
        ]
        return feature_names_in_a, combined_categories


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Relational algebra reimplementation of scikit-learn's `OrdinalEncoder`_ transformer that encodes categorical features as numbers.
Works on both pandas and Spark dataframes by using `Aggregate`_ for `fit` and `Map`_ for `transform`, which in turn use the appropriate backend.

.. _`OrdinalEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
.. _`Aggregate`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.aggregate.html
.. _`Map`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.ordinal_encoder.html",
    "type": "object",
    "tags": {
        "pre": ["categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": ordinal_encoder._hyperparams_schema,
        "input_fit": ordinal_encoder._input_fit_schema,
        "input_transform": ordinal_encoder._input_transform_schema,
        "output_transform": ordinal_encoder._output_transform_schema,
    },
}

OrdinalEncoder = lale.operators.make_operator(_OrdinalEncoderImpl, _combined_schemas)

OrdinalEncoder = typing.cast(
    lale.operators.PlannedIndividualOp,
    OrdinalEncoder.customize_schema(
        encode_unknown_with=None,
        dtype={
            "enum": ["float64"],
            "description": "This implementation only supports `dtype='float64'`.",
            "default": "float64",
        },
        handle_unknown={
            "enum": ["use_encoded_value"],
            "description": "This implementation only supports `handle_unknown='use_encoded_value'`.",
            "default": "use_encoded_value",
        },
        unknown_value={
            "anyOf": [
                {"type": "integer"},
                {"enum": [np.nan, None]},
            ],
            "description": "The encoded value of unknown categories to use when `handle_unknown='use_encoded_value'`. It has to be distinct from the values used to encode any of the categories in fit. If set to np.nan, the dtype hyperparameter must be a float dtype.",
        },
    ),
)

lale.docstrings.set_docstrings(OrdinalEncoder)
