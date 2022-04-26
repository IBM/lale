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

import ast
import typing

import pandas as pd

import lale.docstrings
import lale.operators
from lale.datasets.data_schemas import add_table_name, forward_metadata, get_table_name
from lale.expressions import _it_column
from lale.helpers import (
    _is_ast_call,
    _is_ast_name,
    _is_ast_subs_or_attr,
    _is_pandas_df,
    _is_spark_df,
    _is_spark_with_index,
)
from lale.lib.dataframe import get_columns
from lale.lib.rasl._eval_pandas_df import eval_expr_pandas_df
from lale.lib.rasl._eval_spark_df import eval_expr_spark_df

try:
    # noqa in the imports here because those get used dynamically and flake fails.
    from pyspark.sql.functions import col as spark_col  # noqa

    spark_installed = True
except ImportError:
    spark_installed = False


def _new_column_name(name, expr):
    if name is None or (isinstance(name, str) and not name.strip()):
        return _infer_new_name(expr)
    else:
        return name


def _infer_new_name(expr):
    if (
        _is_ast_call(expr._expr)
        and _is_ast_name(expr._expr.func)
        and expr._expr.func.id
        in [
            "identity",
            "isnan",
            "isnotnan",
            "isnull",
            "isnotnull",
            "replace",
            "day_of_month",
            "day_of_week",
            "day_of_year",
            "hour",
            "minute",
            "month",
        ]
        and _is_ast_subs_or_attr(expr._expr.args[0])
    ):
        return _it_column(expr._expr.args[0])
    else:
        raise ValueError(
            """New name of the column to be renamed cannot be None or empty. You may want to use a dictionary
            to specify the new column name as the key, and the expression as the value."""
        )


def _validate(X, expr):
    visitor = _Validate(X)
    visitor.visit(expr._expr)
    return visitor.accessed


class _Validate(ast.NodeVisitor):
    def __init__(self, X):
        self.df = X
        self.accessed = set()

    def visit_Attribute(self, node: ast.Attribute):
        column_name = _it_column(node)
        if column_name not in self.df.columns:
            raise ValueError(
                f"The column {column_name} is not present in the dataframe"
            )
        self.accessed.add(_it_column(node))

    def visit_Subscript(self, node: ast.Subscript):
        column_name = _it_column(node)
        if column_name is None or (
            isinstance(column_name, str) and not column_name.strip()
        ):
            raise ValueError("Name of the column cannot be None or empty.")
        if column_name not in self.df.columns:
            raise ValueError(
                f"The column {column_name} is not present in the dataframe"
            )
        self.accessed.add(column_name)


class _MapImpl:
    def __init__(self, columns, remainder="drop"):
        self.columns = columns
        self.remainder = remainder

    def fit(self, X, y=None):
        if callable(self.columns):
            self.columns = self.columns(X)
        return self

    def __getattribute__(self, item):
        # we want to remove fit if a static column is available
        # since it should be considered already trained
        omit_fit = False
        if item == "fit":
            try:
                cols = super().__getattribute__("columns")
                if not callable(cols):
                    omit_fit = True
            except AttributeError:
                pass
        if omit_fit:
            raise AttributeError(
                "fit cannot be called on a Map that has a static expression or has already been fit"
            )
        else:
            return super().__getattribute__(item)

    def transform(self, X):
        if _is_pandas_df(X):
            return self.transform_pandas_df(X)
        elif _is_spark_df(X):
            return self.transform_spark_df(X)
        else:
            raise ValueError(
                f"Only Pandas or Spark dataframe are supported as inputs, got {type(X)}. Please check that pyspark is installed if you see this error for a Spark dataframe."
            )

    def transform_pandas_df(self, X):
        mapped_df = {}
        accessed_column_names = set()

        def get_map_function_output(column, new_column_name):
            accessed_columns = _validate(X, column)
            new_column_name = _new_column_name(new_column_name, column)
            new_column = eval_expr_pandas_df(X, column)
            mapped_df[new_column_name] = new_column
            accessed_column_names.add(new_column_name)
            accessed_column_names.update(accessed_columns)

        columns = self.columns
        if callable(columns):
            columns = columns(X)

        if isinstance(columns, list):
            for column in columns:
                get_map_function_output(column, None)
        elif isinstance(columns, dict):
            for new_column_name, column in columns.items():
                get_map_function_output(column, new_column_name)
        else:
            raise ValueError("columns must be either a list or a dictionary.")
        mapped_df = pd.DataFrame(mapped_df)
        if self.remainder == "passthrough":
            remainder_columns = [x for x in X.columns if x not in accessed_column_names]
            mapped_df[remainder_columns] = X[remainder_columns]
        table_name = get_table_name(X)
        mapped_df = add_table_name(mapped_df, table_name)
        return mapped_df

    def transform_spark_df(self, X):
        new_columns = []
        accessed_column_names = set()

        def get_map_function_expr(column, new_column_name):
            accessed_columns = _validate(X, column)
            new_column_name = _new_column_name(new_column_name, column)
            new_column = eval_expr_spark_df(column)  # type: ignore
            new_columns.append(new_column.alias(new_column_name))  # type: ignore
            accessed_column_names.add(new_column_name)
            accessed_column_names.update(accessed_columns)

        columns = self.columns
        if callable(columns):
            columns = columns(X)

        if isinstance(columns, list):
            for column in columns:
                get_map_function_expr(column, None)
        elif isinstance(columns, dict):
            for new_column_name, column in columns.items():
                get_map_function_expr(column, new_column_name)
        else:
            raise ValueError("columns must be either a list or a dictionary.")
        if self.remainder == "passthrough":
            remainder_columns = [
                spark_col(typing.cast(str, x))
                for x in get_columns(X)
                if x not in accessed_column_names
            ]
            new_columns.extend(remainder_columns)
        if _is_spark_with_index(X):
            for index_name in X.index_names:
                if index_name not in accessed_column_names:
                    new_columns.extend([spark_col(index_name)])
        mapped_df = X.select(new_columns)
        mapped_df = forward_metadata(X, mapped_df)
        return mapped_df


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
                        {
                            "description": "A callable which, when given the input data, returns either a list or dictionary of mapping expressions, as above.",
                            "laleType": "callable",
                            "forOptimizer": False,
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
                    "default": "drop",
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
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "input_fit": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Map = lale.operators.make_operator(_MapImpl, _combined_schemas)

lale.docstrings.set_docstrings(Map)
