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
import collections
import importlib
from typing import Any

import pandas as pd

from lale.expressions import AstExpr, Expr, _it_column
from lale.helpers import _ast_func_id


def eval_expr_pandas_df(X, expr: Expr) -> pd.Series:
    return _eval_ast_expr_pandas_df(X, expr._expr)


def _eval_ast_expr_pandas_df(X, expr: AstExpr) -> pd.Series:
    evaluator = _PandasEvaluator(X)
    evaluator.visit(expr)
    return evaluator.result


class _PandasEvaluator(ast.NodeVisitor):
    def __init__(self, X):
        self.result = None
        self.df = X

    def visit_Num(self, node: ast.Num):
        self.result = node.n

    def visit_Str(self, node: ast.Str):
        self.result = node.s

    def visit_Constant(self, node: ast.Constant):
        self.result = node.value

    def visit_Attribute(self, node: ast.Attribute):
        column_name = _it_column(node)
        self.result = self.df[column_name]

    def visit_Subscript(self, node: ast.Subscript):
        column_name = _it_column(node)
        self.result = self.df[column_name]

    def visit_BinOp(self, node: ast.BinOp):
        self.visit(node.left)
        v1 = self.result
        self.visit(node.right)
        v2 = self.result
        if isinstance(node.op, ast.Add):
            self.result = v1 + v2  # type: ignore
        elif isinstance(node.op, ast.Sub):
            self.result = v1 - v2  # type: ignore
        elif isinstance(node.op, ast.Mult):
            self.result = v1 * v2  # type: ignore
        elif isinstance(node.op, ast.Div):
            self.result = v1 / v2  # type: ignore
        elif isinstance(node.op, ast.FloorDiv):
            self.result = v1 // v2  # type: ignore
        elif isinstance(node.op, ast.Mod):
            self.result = v1 % v2  # type: ignore
        elif isinstance(node.op, ast.Pow):
            self.result = v1 ** v2  # type: ignore
        else:
            raise ValueError(f"""Unimplemented operator {ast.dump(node.op)}""")

    def visit_Call(self, node: ast.Call):
        functions_module = importlib.import_module("lale.lib.rasl._eval_pandas_df")
        function_name = _ast_func_id(node.func)
        map_func_to_be_called = getattr(functions_module, function_name)
        self.result = map_func_to_be_called(self.df, node)


def replace(df: Any, call: ast.Call):
    column = _eval_ast_expr_pandas_df(df, call.args[0])  # type: ignore
    mapping_dict = ast.literal_eval(call.args[1].value)  # type: ignore
    handle_unknown = ast.literal_eval(call.args[2])
    if handle_unknown == "use_encoded_value":
        unknown_value = ast.literal_eval(call.args[3])
        mapping2 = collections.defaultdict(lambda: unknown_value, mapping_dict)
        new_column = column.map(mapping2)  # type: ignore
    else:
        new_column = column.replace(mapping_dict)
    return new_column


def identity(df: Any, call: ast.Call):
    return _eval_ast_expr_pandas_df(df, call.args[0])  # type: ignore


def time_functions(df: Any, call, pandas_func: str):
    fmt = None
    column = _eval_ast_expr_pandas_df(df, call.args[0])
    if len(call.args) > 1:
        fmt = ast.literal_eval(call.args[1])
    new_column = pd.to_datetime(column, format=fmt)
    return getattr(getattr(new_column, "dt"), pandas_func)


def day_of_month(df: Any, call: ast.Call):
    return time_functions(df, call, "day")


def day_of_week(df: Any, call: ast.Call):
    return time_functions(df, call, "weekday")


def day_of_year(df: Any, call: ast.Call):
    return time_functions(df, call, "dayofyear")


def hour(df: Any, call: ast.Call):
    return time_functions(df, call, "hour")


def minute(df: Any, call: ast.Call):
    return time_functions(df, call, "minute")


def month(df: Any, call: ast.Call):
    return time_functions(df, call, "month")
