# Copyright 2022 IBM Corporation
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

import lale.docstrings
import lale.operators
from lale.datasets.data_schemas import forward_metadata, get_index_names
from lale.helpers import _is_pandas_df, _is_pandas_series, _is_spark_df


class _SortIndexImpl:
    def __init__(self, ascending=True):
        self.ascending = ascending

    def transform_schema(self, s_X):
        return s_X

    def transform(self, X):
        if _is_pandas_df(X):
            ordered_df = X.sort_index(ascending=self.ascending)
        elif _is_spark_df(X):
            index_cols = get_index_names(X)  # type:ignore
            ordered_df = X.orderBy(index_cols, ascending=self.ascending)
        else:
            raise ValueError(
                "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
            )
        ordered_df = forward_metadata(X, ordered_df)
        return ordered_df

    def transform_X_y(self, X, y=None):
        result_y = None
        if y is not None:
            assert _is_pandas_df(y) or _is_pandas_series(
                y
            ), "transform_X_y is supported only when y is a Pandas Series or DataFrame."
            result_y = y.sort_index(
                ascending=self.ascending
            )  # assumes that y is always Pandas
        result_X = self.transform(X)
        return result_X, result_y


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["ascending"],
            "relevantToOptimizer": [],
            "properties": {
                "ascending": {
                    "description": "Sort by index of the dataframe.",
                    "type": "boolean",
                    "default": True,
                }
            },
        }
    ]
}


_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "The outer array is over rows.",
            "anyOf": [
                {"laleType": "Any"},
                {
                    "type": "array",
                    "items": {
                        "description": "The inner array is over columns.",
                        "type": "array",
                        "items": {"laleType": "Any"},
                    },
                },
            ],
        }
    },
}

_output_transform_schema = {
    "description": "The outer array is over rows.",
    "anyOf": [
        {
            "type": "array",
            "items": {
                "description": "The inner array is over columns.",
                "type": "array",
                "items": {"laleType": "Any"},
            },
        },
        {"laleType": "Any"},
    ],
}

_input_transform_X_y_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input features as numpy, pandas, or PySpark.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        "y": {
            "anyOf": [
                {"enum": [None]},
                {
                    "description": "Input labels as numpy, pandas, or PySpark.",
                    "type": "array",
                    "items": {"laleType": "Any"},
                },
            ],
        },
    },
}

_output_transform_X_y_schema = {
    "type": "array",
    "laleType": "tuple",
    "items": [
        {
            "description": "X",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        {
            "anyOf": [
                {"enum": [None]},
                {
                    "description": "Input labels as numpy, pandas, or PySpark.",
                    "type": "array",
                    "items": {"laleType": "Any"},
                },
            ],
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "SortIndex operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.sort_index.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_transform_X_y": _input_transform_X_y_schema,
        "output_transform_X_y": _output_transform_X_y_schema,
    },
}


SortIndex = lale.operators.make_operator(_SortIndexImpl, _combined_schemas)

lale.docstrings.set_docstrings(SortIndex)
