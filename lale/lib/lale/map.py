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

import lale.docstrings
import lale.operators


class MapImpl:
    def __init__(self, columns, remainder="passthrough"):
        self.columns = columns
        self.remainder = remainder

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.remainder == "passthrough":
            out_df = X
        elif self.remainder == "drop":
            out_df = pd.DataFrame()
        else:
            raise ValueError("remainder has to be either `passthrough` or `drop`.")

        def get_map_function_output(column):
            functions_module = importlib.import_module("lale.lib.lale.functions")
            function_name = column._expr.func.id
            map_func_to_be_called = getattr(functions_module, function_name)
            return map_func_to_be_called(X, column)

        if isinstance(self.columns, list):
            for column in self.columns:
                column_name, new_column = get_map_function_output(column)
                # Since this is a list, we have to use the column_name from the function output
                out_df[column_name] = new_column
        elif isinstance(self.columns, dict):
            for new_column_name, column in self.columns.items():
                column_name, new_column = get_map_function_output(column)
                out_df[new_column_name] = new_column
                del out_df[column_name]
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
            "type": "array",
            "items": {
                "description": "The inner array is over columns.",
                "type": "array",
                "items": {"laleType": "Any"},
            },
        }
    },
}

_output_transform_schema = {
    "description": "The outer array is over rows.",
    "type": "array",
    "items": {
        "description": "The inner array is over columns.",
        "type": "array",
        "items": {"laleType": "Any"},
    },
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

lale.docstrings.set_docstrings(MapImpl, _combined_schemas)

Map = lale.operators.make_operator(MapImpl, _combined_schemas)
