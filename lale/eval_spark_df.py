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
from itertools import chain

from lale.expressions import AstExpr, Expr
from lale.helpers import _ast_func_id, _is_ast_name_it

try:
    # noqa in the imports here because those get used dynamically and flake fails.
    from pyspark.sql import Column  # noqa
    from pyspark.sql.functions import col  # noqa
    from pyspark.sql.functions import lit  # noqa
    from pyspark.sql.functions import to_timestamp  # noqa
    from pyspark.sql.functions import hour as spark_hour  # noqa
    from pyspark.sql.functions import minute as spark_minute  # noqa
    from pyspark.sql.functions import month as spark_month  # noqa

    from pyspark.sql.functions import (  # noqa; isort: skip
        create_map,
        dayofmonth,
        dayofweek,
        dayofyear,
        floor as spark_floor,
    )

    spark_installed = True
except ImportError:
    spark_installed = False


def eval_expr_spark_df(expr: Expr) -> Column:
    return _eval_ast_expr_spark_df(expr._expr)


def _eval_ast_expr_spark_df(expr: AstExpr) -> Column:
    evaluator = _SparkEvaluator()
    evaluator.visit(expr)
    return evaluator.result


class _SparkEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.result = None

    def visit_Num(self, node: ast.Num):
        self.result = lit(node.n)

    def visit_Str(self, node: ast.Str):
        self.result = lit(node.s)

    def visit_Constant(self, node: ast.Constant):
        self.result = lit(node.value)

    def visit_Subscript(self, node: ast.Subscript):
        if _is_ast_name_it(node.value):
            column_name = node.slice.value.s  # type: ignore
            if column_name is None or not column_name.strip():
                raise ValueError("Name of the column cannot be None or empty.")
            self.result = col(column_name)
        else:
            raise ValueError("Unimplemented expression")

    def visit_Attribute(self, node: ast.Attribute):
        if _is_ast_name_it(node.value):
            self.result = col(node.attr)
        else:
            raise ValueError("Unimplemented expression")

    def visit_BinOp(self, node: ast.BinOp):
        self.visit(node.left)
        v1 = self.result
        self.visit(node.right)
        v2 = self.result
        assert v1 is not None
        assert v2 is not None
        if isinstance(node.op, ast.Add):
            self.result = v1 + v2
        elif isinstance(node.op, ast.Sub):
            self.result = v1 - v2
        elif isinstance(node.op, ast.Mult):
            self.result = v1 * v2
        elif isinstance(node.op, ast.Div):
            self.result = v1 / v2
        elif isinstance(node.op, ast.FloorDiv):
            self.result = spark_floor(v1 / v2)
        elif isinstance(node.op, ast.Mod):
            self.result = v1 % v2
        elif isinstance(node.op, ast.Pow):
            self.result = v1 ** v2
        else:
            raise ValueError(f"""Unimplemented operator {ast.dump(node.op)}""")

    def visit_Call(self, node: ast.Call):
        functions_module = importlib.import_module("lale.eval_spark_df")
        function_name = _ast_func_id(node.func)
        map_func_to_be_called = getattr(functions_module, function_name)
        self.result = map_func_to_be_called(node)


def replace(call: ast.Call):
    column = _eval_ast_expr_spark_df(call.args[0])  # type: ignore
    mapping_dict = ast.literal_eval(call.args[1].value)  # type: ignore
    mapping_expr = create_map([lit(x) for x in chain(*mapping_dict.items())])  # type: ignore
    return mapping_expr[column]


def identity(call: ast.Call):
    return _eval_ast_expr_spark_df(call.args[0])  # type: ignore


def time_functions(call, spark_func):
    column = _eval_ast_expr_spark_df(call.args[0])
    if len(call.args) > 1:
        fmt = ast.literal_eval(call.args[1])
        return spark_func(to_timestamp(column, format=fmt))
    return spark_func(to_timestamp(column))


def day_of_month(call: ast.Call):
    return time_functions(call, dayofmonth)


def day_of_week(call: ast.Call):
    return time_functions(call, dayofweek)


def day_of_year(call: ast.Call):
    return time_functions(call, dayofyear)


def hour(call: ast.Call):
    return time_functions(call, spark_hour)


def minute(call: ast.Call):
    return time_functions(call, spark_minute)


def month(call: ast.Call):
    return time_functions(call, spark_month)
