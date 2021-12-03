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

import ast
import importlib
from typing import Any

import pandas as pd

from lale.expressions import AstExpr
from lale.helpers import _is_ast_attribute, _is_ast_name, _is_ast_subscript


def eval_pandas_df(X, expr):
    evaluator = _PandasEvaluator(X)
    evaluator.visit(expr._expr)
    return evaluator.result


class _PandasEvaluator(ast.NodeVisitor):
    def __init__(self, X):
        self.result = None
        self.df = X

    def visit_Constant(self, node: ast.Constant):
        self.result = node.value

    def visit_Subscript(self, node: ast.Subscript):
        if _is_ast_name(node.value) and node.value.id == "it":
            self.visit(node.slice)
            column_name = self.result
            if column_name is None or not column_name.strip():
                raise ValueError("Name of the column cannot be None or empty.")
            self.result = self.df[column_name]
        else:
            raise ValueError("Unimplemented expression")

    def visit_Attribute(self, node: ast.Attribute):
        if _is_ast_name(node.value) and node.value.id == "it":
            self.result = self.df[node.attr]
        else:
            raise ValueError("Unimplemented expression")

    def visit_BinOp(self, node: ast.BinOp):
        self.visit(node.left)
        v1 = self.result
        self.visit(node.right)
        v2 = self.result
        if isinstance(node.op, ast.Add):
            self.result = v1 + v2
        elif isinstance(node.op, ast.Sub):
            self.result = v1 - v2
        elif isinstance(node.op, ast.Mult):
            self.result = v1 * v2
        elif isinstance(node.op, ast.Div):
            self.result = v1 / v2
        elif isinstance(node.op, ast.FloorDiv):
            self.result = v1 // v2
        elif isinstance(node.op, ast.Mod):
            self.result = v1 % v2
        elif isinstance(node.op, ast.Pow):
            self.result = v1 ** v2
        else:
            raise ValueError(f"""Unimplemented operator {ast.dump(node.op)}""")

    def visit_Call(self, node: ast.Call):
        functions_module = importlib.import_module("lale.eval_pandas_df")
        function_name = node.func.id
        map_func_to_be_called = getattr(functions_module, function_name)
        self.result = map_func_to_be_called(self.df, node)


def replace(df: Any, replace_expr: AstExpr):
    column_name = replace_expr.args[0].attr
    mapping_dict = ast.literal_eval(replace_expr.args[1].value)
    new_column = df[column_name].replace(mapping_dict)
    return new_column


def identity(df: Any, column: AstExpr):
    if _is_ast_subscript(column):  # type: ignore
        column_name = column.slice.value.s  # type: ignore
    elif _is_ast_attribute(column):  # type: ignore
        column_name = column.attr  # type: ignore
    else:
        raise ValueError(
            "Expression type not supported. Formats supported: it.column_name or it['column_name']."
        )
    return df[column_name]


def ratio(df: Any, expr: AstExpr):
    numerator = eval_pandas_df(df, expr.args[0])  # type: ignore
    denominator = eval_pandas_df(df, expr.args[1])  # type: ignore
    return numerator / denominator


def subtract(df: Any, expr: AstExpr):
    e1 = eval_pandas_df(df, expr.args[0])  # type: ignore
    e2 = eval_pandas_df(df, expr.args[1])  # type: ignore
    return e1 / e2


def time_functions(df: Any, dom_expr: AstExpr, pandas_func: str):
    fmt = None
    column_name = dom_expr.args[0].attr
    if len(dom_expr.args) > 1:
        fmt = ast.literal_eval(dom_expr.args[1])
    new_column = pd.to_datetime(df[column_name], format=fmt)
    return getattr(getattr(new_column, "dt"), pandas_func)


def day_of_month(df: Any, dom_expr: AstExpr):
    return time_functions(df, dom_expr, "day")


def day_of_week(df: Any, dom_expr: AstExpr):
    return time_functions(df, dom_expr, "weekday")


def day_of_year(df: Any, dom_expr: AstExpr):
    return time_functions(df, dom_expr, "dayofyear")


def hour(df: Any, dom_expr: AstExpr):
    return time_functions(df, dom_expr, "hour")


def minute(df: Any, dom_expr: AstExpr):
    return time_functions(df, dom_expr, "minute")


def month(df: Any, dom_expr: AstExpr):
    return time_functions(df, dom_expr, "month")


def string_indexer(df: pd.DataFrame, dom_expr: AstExpr):
    column_name = dom_expr.args[0].attr
    sorted_indices = df[column_name].value_counts().index
    new_column = df[column_name].map(
        dict(zip(sorted_indices, range(0, len(sorted_indices))))
    )
    return new_column
