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

import pandas as pd

import lale.docstrings
import lale.operators

try:
    import pyspark
    from pyspark.sql.dataframe import DataFrame as spark_df

    spark_installed = True

except ImportError:
    spark_installed = False


def _is_pandas_df(df):
    return isinstance(df, pd.core.groupby.generic.DataFrameGroupBy) or isinstance(
        df, pd.DataFrame
    )


def _is_spark_df(df):
    return isinstance(df, pyspark.sql.group.GroupedData) or isinstance(df, spark_df)  # type: ignore


def _is_ast_subscript(expr):
    return isinstance(expr, ast.Subscript)


def _is_ast_attribute(expr):
    return isinstance(expr, ast.Attribute)


class _AggregateImpl:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        if _is_pandas_df(X):
            agg_col = []
            agg_func = []
            rename_col = []
            agg_expr = {}

            for new_col_name, expr in self.columns.items():
                rename_col.append(new_col_name)
                agg_func.append(expr._expr.func.id)
                expr_to_parse = expr._expr.args[0]
                if _is_ast_subscript(expr_to_parse):
                    agg_col.append(expr_to_parse.slice.value.s)  # type: ignore
                elif _is_ast_attribute(expr_to_parse):
                    agg_col.append(expr_to_parse.attr)
                else:
                    raise ValueError(
                        "Aggregate 'columns' parameter only supports subscript or dot notation for the key columns. For example, it.col_name or it['col_name']."
                    )

            zipped_lists = zip(agg_col, agg_func, rename_col)
            sorted_pairs = sorted(zipped_lists)
            tuples = zip(*sorted_pairs)
            agg_col, agg_func, rename_col = [list(tuple) for tuple in tuples]

            for col, func in zip(agg_col, agg_func):
                if col in agg_expr:
                    agg_expr[col].append(func)
                else:
                    agg_expr[col] = [func]
            try:
                aggregated_df = X.agg(agg_expr)
                aggregated_df.columns = rename_col
                return aggregated_df
            except KeyError as e:
                raise KeyError(e)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "columns": {
                    "description": "Aggregations for producing output columns.",
                    "anyOf": [
                        {
                            "description": "Dictionary of output column names and aggregation expressions.",
                            "type": "object",
                            "additionalProperties": {"laleType": "expression"},
                        },
                        {
                            "description": "List of aggregation expressions. The output column name is determined by a heuristic based on the input column name and the transformation function.",
                            "type": "array",
                            "items": {"laleType": "expression"},
                        },
                    ],
                    "default": [],
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
            "description": "List of tables.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
            "minItems": 1,
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
    "description": "Relational algebra aggregate operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.aggregate.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Aggregate = lale.operators.make_operator(_AggregateImpl, _combined_schemas)

lale.docstrings.set_docstrings(Aggregate)
