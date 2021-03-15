# Copyright 2020 IBM Corporation
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
import importlib

import pandas as pd

try:
    from pyspark.sql.dataframe import DataFrame as spark_df

    spark_installed = True
except ImportError:
    spark_installed = False

import lale.docstrings
import lale.operators


class _MapImpl:
    def __init__(self, columns, remainder="passthrough"):
        self.columns = columns
        self.remainder = remainder

    def transform(self, X):
        columns_to_keep = []

        def get_map_function_output(column, new_column_name):
            functions_module = importlib.import_module("lale.lib.lale.functions")
            function_name = column._expr.func.id
            map_func_to_be_called = getattr(functions_module, function_name)
            return map_func_to_be_called(X, column, new_column_name)

        if isinstance(self.columns, list):
            for column in self.columns:
                new_column_name, X = get_map_function_output(column, None)
                columns_to_keep.append(new_column_name)
        elif isinstance(self.columns, dict):
            for new_column_name, column in self.columns.items():
                new_column_name, X = get_map_function_output(column, new_column_name)
                columns_to_keep.append(new_column_name)
        else:
            raise ValueError("columns must be either a list or a dictionary.")

        out_df = X  # Do nothing as X already has the right columns
        if self.remainder == "drop":
            if isinstance(X, pd.DataFrame):
                out_df = X[columns_to_keep]
            elif spark_installed and isinstance(X, spark_df):
                out_df = X.select(columns_to_keep)
            else:
                raise ValueError(
                    "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
                )
        return out_df


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "columns": {
                    "description": "Mappings for producing output columns.",
                    "anyOf": [
                        {
                            "description": "Dictionary of output column names and mapping expressions.",
                            "type": "object",
                            "additionalProperties": {"laleType": "expression"},
                        },
                        {
                            "description": "List of mapping expressions. The output column name is determined by a heuristic based on the input column name and the transformation function.",
                            "type": "array",
                            "items": {"laleType": "expression"},
                        },
                    ],
                    "default": [],
                },
                "remainder": {
                    "description": "Transformation for the remaining columns.",
                    "anyOf": [
                        {"enum": ["passthrough", "drop"]},
                        {"description": "Mapping expression.", "laleType": "operator"},
                    ],
                    "default": "passthrough",
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

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra map operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.map.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Map = lale.operators.make_operator(_MapImpl, _combined_schemas)

lale.docstrings.set_docstrings(Map)
