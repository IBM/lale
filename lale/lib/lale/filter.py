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
    from pyspark.sql.dataframe import DataFrame as spark_df
    from pyspark.sql.functions import col

    spark_installed = True

except ImportError:
    spark_installed = False


def _is_df(df):
    return isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)


def _is_spark_df(df):
    return isinstance(df, spark_df)


def _is_ast_subscript(expr):
    return isinstance(expr, ast.Subscript)


class _FilterImpl:
    def __init__(self, pred=None):
        self.pred = pred

    # Parse the predicate element passed as input
    def _get_filter_info(self, expr_to_parse, X):
        col_list = X.columns
        lhs = expr_to_parse.left.slice.value.s
        if lhs not in col_list:
            raise ValueError(
                "Cannot perform filter operation as {} not a column of input dataframe X.".format(
                    lhs
                )
            )
        op = expr_to_parse.ops[0]
        rhs = expr_to_parse.comparators[0]
        if _is_ast_subscript(expr_to_parse.comparators[0]):
            rhs = expr_to_parse.comparators[0].slice.value.s
            if rhs not in col_list:
                raise ValueError(
                    "Cannot perform filter operation as {} not a column of input dataframe X.".format(
                        rhs
                    )
                )
        return lhs, op, rhs

    def transform(self, X):
        filtered_df = X

        def filter(X):
            # Filtering spark dataframes
            if _is_spark_df(X):
                if isinstance(op, ast.Eq):
                    return (
                        X.filter(col(lhs) == col(rhs))
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X.filter(col(lhs) == rhs)
                    )
                elif isinstance(op, ast.NotEq):
                    return (
                        X.filter(col(lhs) != col(rhs))
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X.filter(col(lhs) != rhs)
                    )
                elif isinstance(op, ast.GtE):
                    return (
                        X.filter(col(lhs) >= col(rhs))
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X.filter(col(lhs) >= rhs)
                    )
                elif isinstance(op, ast.Gt):
                    return (
                        X.filter(col(lhs) > col(rhs))
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X.filter(col(lhs) > rhs)
                    )
                elif isinstance(op, ast.LtE):
                    return (
                        X.filter(col(lhs) <= col(rhs))
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X.filter(col(lhs) <= rhs)
                    )
                elif isinstance(op, ast.Lt):
                    return (
                        X.filter(col(lhs) < col(rhs))
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X.filter(col(lhs) < rhs)
                    )
                else:
                    raise ValueError(
                        "{} operator type found. Only ==, !=, >=, <=, >, < operators supported.".format(
                            op
                        )
                    )
            # Filtering pandas dataframes
            if _is_df(X):
                if isinstance(op, ast.Eq):
                    return (
                        X[X[lhs] == X[rhs]]
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X[X[lhs] == rhs]
                    )
                elif isinstance(op, ast.NotEq):
                    return (
                        X[X[lhs] != X[rhs]]
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X[X[lhs] != rhs]
                    )
                elif isinstance(op, ast.GtE):
                    return (
                        X[X[lhs] >= X[rhs]]
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X[X[lhs] >= rhs]
                    )
                elif isinstance(op, ast.Gt):
                    return (
                        X[X[lhs] > X[rhs]]
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X[X[lhs] > rhs]
                    )
                elif isinstance(op, ast.LtE):
                    return (
                        X[X[lhs] <= X[rhs]]
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X[X[lhs] <= rhs]
                    )
                elif isinstance(op, ast.Lt):
                    return (
                        X[X[lhs] < X[rhs]]
                        if _is_ast_subscript(expr_to_parse.comparators[0])
                        else X[X[lhs] < rhs]
                    )
                else:
                    raise ValueError(
                        "{} operator type found. Only ==, !=, >=, <=, >, < operators supported.".format(
                            op
                        )
                    )
            else:
                raise ValueError(
                    "Only pandas and spark dataframes are supported by the filter operator."
                )

        for pred_element in self.pred if self.pred is not None else []:
            expr_to_parse = pred_element._expr
            lhs, op, rhs = self._get_filter_info(expr_to_parse, X)
            filtered_df = filter(filtered_df)
        return filtered_df


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["pred"],
            "relevantToOptimizer": [],
            "properties": {
                "pred": {
                    "description": "Filter predicate. Given as Python AST expression.",
                    "laleType": "Any",
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
            "description": "Input table or dataframe",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
            "minItems": 1,
        }
    },
}

_output_transform_schema = {
    "description": "Features; no restrictions on data type.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra filter operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.filter.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Filter = lale.operators.make_operator(_FilterImpl, _combined_schemas)

lale.docstrings.set_docstrings(Filter)
