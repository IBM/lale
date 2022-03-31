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

import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators
from lale.datasets import pandas2spark
from lale.datasets.data_schemas import (
    SparkDataFrameWithIndex,
    add_table_name,
    get_table_name,
)
from lale.helpers import _is_spark_df, _is_spark_with_index


def _convert(data, astype, X_or_y):
    if _is_spark_df(data):
        if astype == "pandas":
            if X_or_y == "X":
                result = data.toPandas()
            else:
                result = data.toPandas().squeeze()
        elif astype == "spark":
            if _is_spark_with_index(data):
                result = data.drop_indexes()
            else:
                result = data
        elif astype == "spark-with-index":
            if _is_spark_with_index(data):
                result = data
            else:
                result = SparkDataFrameWithIndex(data)
        else:
            assert False, astype
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        if astype == "pandas":
            result = data
        elif astype == "spark":
            result = pandas2spark(data)
        elif astype == "spark-with-index":
            result = pandas2spark(data, with_index=True)
        else:
            assert False, astype
    elif isinstance(data, np.ndarray):
        if astype == "pandas":
            if X_or_y == "X":
                result = pd.DataFrame(data)
            else:
                result = pd.Series(data)
        elif astype == "spark":
            result = pandas2spark(pd.DataFrame(data))
        elif astype == "spark-with-index":
            result = pandas2spark(pd.DataFrame(data), with_index=True)
        else:
            assert False, astype
    else:
        raise TypeError(f"unexpected type {type(data)}")
    result = add_table_name(result, get_table_name(data))
    return result


class _ConvertImpl:
    def __init__(self, astype="pandas"):
        self.astype = astype

    def transform(self, X):
        return _convert(X, self.astype, "X")

    def transform_X_y(self, X, y):
        result_X = self.transform(X)
        result_y = None if y is None else _convert(y, self.astype, "y")
        return result_X, result_y

    def viz_label(self) -> str:
        return "Convert:\n" + self.astype


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["astype"],
            "relevantToOptimizer": [],
            "properties": {
                "astype": {
                    "description": "Type to convert to.",
                    "enum": ["pandas", "spark", "spark-with-index"],
                    "default": "pandas",
                },
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
            "description": "Input features as numpy, pandas, or PySpark.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        }
    },
}

_output_transform_schema = {
    "type": "array",
    "items": {"type": "array", "items": {"laleType": "Any"}},
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
    "description": "Convert data to different representation if necessary.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.convert.html",
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


Convert = lale.operators.make_operator(_ConvertImpl, _combined_schemas)

lale.docstrings.set_docstrings(Convert)
