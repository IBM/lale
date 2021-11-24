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

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
from lale.helpers import (
    _is_ast_call,
    _is_ast_name,
    _is_pandas_df,
    _is_spark_df,
    pandas_df_eval,
)

import pandas as pd

def _new_column_name(name, expr):
    def infer_new_name(expr):
        if (
            _is_ast_call(expr._expr) and
            _is_ast_name(expr._expr.func) and
            expr._expr.func.id in [ "replace",
                                    "day_of_month",
                                    "day_of_week",
                                    "day_of_year",
                                    "hour",
                                    "minute",
                                    "month",
                                    "string_indexer" ]
        ):
            return expr._expr.args[0].attr
        else: 
            raise ValueError(
                """New name of the column to be renamed cannot be None or empty. You may want to use a dictionary
                to specify the new column name as the key, and the expression as the value."""
            )
    if name is None or not name.strip():
        return infer_new_name(expr)
    else:
        return name

class _MapImpl:
    def __init__(self, columns, remainder="drop"):
        self.columns = columns
        self.remainder = remainder

    def transform(self, X):
        is_pandas = False
        is_spark = False
        table_name = lale.datasets.data_schemas.get_table_name(X)
        if _is_pandas_df(X):
            is_pandas = True
            mapped_df = pd.DataFrame()
        elif _is_spark_df(X):
            is_spark = True
            mapped_df = None # TODO
        else:
            raise ValueError(
                "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
            )

        def get_map_function_output(column, new_column_name):
            new_column_name = _new_column_name(new_column_name, column)
            if is_pandas:
                new_column = pandas_df_eval(X, column)
            elif is_spark:
                new_column = None # XXX TODO XXX
            else:
                assert False
            mapped_df[new_column_name] = new_column

        if isinstance(self.columns, list):
            for column in self.columns:
                get_map_function_output(column, None)
        elif isinstance(self.columns, dict):
            for new_column_name, column in self.columns.items():
                get_map_function_output(column, new_column_name)
        else:
            raise ValueError("columns must be either a list or a dictionary.")
        if self.remainder == "passthrough":
            remainder_columns = [] # XXX TODO XXX
            if _is_pandas_df(X):
                for columns in remainder_columns:
                    mapped_df[column] = X[column]
            elif _is_spark_df(X):
                for columns in remainder_columns:
                    pass # XXX TODO XXX
            else:
                raise ValueError(
                    "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
                )
        mapped_df = lale.datasets.data_schemas.add_table_name(mapped_df, table_name)
        return mapped_df


# class _MapImpl:
#     def __init__(self, columns, remainder="passthrough"):
#         self.columns = columns
#         self.remainder = remainder

#     def transform(self, X):
#         table_name = lale.datasets.data_schemas.get_table_name(X)
#         columns_to_keep = []

#         def get_map_function_output(column, new_column_name):
#             functions_module = importlib.import_module("lale.lib.lale.functions")
#             if _is_ast_subscript(column._expr) or _is_ast_attribute(column._expr):
#                 function_name = "identity"
#             else:
#                 function_name = column._expr.func.id
#             map_func_to_be_called = getattr(functions_module, function_name)
#             return map_func_to_be_called(X, column, new_column_name)

#         if isinstance(self.columns, list):
#             for column in self.columns:
#                 new_column_name, X = get_map_function_output(column, None)
#                 columns_to_keep.append(new_column_name)
#         elif isinstance(self.columns, dict):
#             for new_column_name, column in self.columns.items():
#                 new_column_name, X = get_map_function_output(column, new_column_name)
#                 columns_to_keep.append(new_column_name)
#         else:
#             raise ValueError("columns must be either a list or a dictionary.")
#         mapped_df = X  # Do nothing as X already has the right columns
#         if self.remainder == "drop":
#             if _is_pandas_df(X):
#                 mapped_df = X[columns_to_keep]
#             elif _is_spark_df(X):
#                 mapped_df = X.select(columns_to_keep)
#             else:
#                 raise ValueError(
#                     "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
#                 )
#         mapped_df = lale.datasets.data_schemas.add_table_name(mapped_df, table_name)
#         return mapped_df


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
