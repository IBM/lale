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
        categories="auto",
        dtype="float64",
        handle_unknown="error",
        unknown_value=None,
    ):
        self.categories = categories
        if categories != "auto":
            self.categories_ = categories
        if dtype != "float64":
            raise ValueError("This implementation only supports `dtype='float64'`.")
        self.dtype = dtype
        if handle_unknown != "use_encoded_value":
            raise ValueError(
                "This implementation only supports `handle_unknown='use_encoded_value'`."
            )
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        # learn the coefficients
        if self.categories == "auto":
            agg_op = Aggregate(columns={c: collect_set(it[c]) for c in X.columns})
            agg_data = agg_op.transform(X)
            if lale.helpers._is_spark_df(agg_data):
                agg_data = agg_data.toPandas()
            self.categories_ = [np.sort(agg_data.loc[0, c]) for c in agg_data.columns]
        # prepare the transformer
        self.transformer = Map(
            columns={
                col_name: replace(
                    it[col_name],
                    {
                        cat_value: cat_idx
                        for cat_idx, cat_value in enumerate(self.categories_[col_idx])
                    },
                    handle_unknown="use_encoded_value",
                    unknown_value=self.unknown_value,
                )
                for col_idx, col_name in enumerate(X.columns)
            }
        )
        return self

    def transform(self, X):
        return self.transformer.transform(X)


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
        "pre": ["~categoricals"],
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
