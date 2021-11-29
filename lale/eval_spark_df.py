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

from lale.expressions import AstExpr
from lale.helpers import (
    _is_ast_attribute,
    _is_ast_subscript,
    _is_ast_name,
)

try:
    from pyspark.ml.feature import StringIndexer

    # noqa in the imports here because those get used dynamically and flake fails.
    from pyspark.sql.functions import hour as spark_hour  # noqa
    from pyspark.sql.functions import lit  # noqa
    from pyspark.sql.functions import col  # noqa
    from pyspark.sql.functions import to_timestamp  # noqa
    from pyspark.sql.functions import minute as spark_minute  # noqa
    from pyspark.sql.functions import month as spark_month  # noqa

    from pyspark.sql.functions import (  # noqa; isort: skip
        create_map,
        dayofmonth,
        dayofweek,
        dayofyear,
    )

    spark_installed = True
except ImportError:
    spark_installed = False


def eval_spark_df(X, expr):
    evaluator = _SparkEvaluator()
    evaluator.visit(expr._expr)
    return evaluator.result

class _SparkEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.result = None

    def visit_Constant(self, node: ast.Constant):
        self.result = lit(node.value)

    def visit_Subscript(self, node: ast.Subscript):
        if _is_ast_name (node.value) and node.value.id == "it":
            column_name = node.slice.value.s  # type: ignore
            if column_name is None or not column_name.strip():
                raise ValueError("Name of the column cannot be None or empty.")
            self.result = col(column_name)
        else:
            raise ValueError(
                f"Unimplemented expression"
            )

    def visit_Attribute(self, node: ast.Attribute):
        if _is_ast_name (node.value) and node.value.id == "it":
            self.result = col(node.attr)
        else:
            raise ValueError(
                f"Unimplemented expression"
            )

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
        # elif isinstance(node.op, ast.FloorDiv):
        #     self.result = v1 // v2
        elif isinstance(node.op, ast.Mod):
            self.result = v1 % v2
        elif isinstance(node.op, ast.Pow):
            self.result = v1 ** v2
        else:
            raise ValueError(
                f"""Unimplemented operator {ast.dump(node.op)}"""
            )

    def visit_Call(self, node: ast.Call):
        functions_module = importlib.import_module("lale.eval_spark_df")
        function_name = node.func.id
        map_func_to_be_called = getattr(functions_module, function_name)
        self.result = map_func_to_be_called(node)


def replace(replace_expr: AstExpr):
    column_name = replace_expr.args[0].attr
    mapping_dict = ast.literal_eval(replace_expr.args[1].value)
    mapping_expr = create_map([lit(x) for x in chain(*mapping_dict.items())])  # type: ignore
    return mapping_expr[col(column_name)]  # type: ignore


def identity(column: AstExpr):
    if _is_ast_subscript(column):  # type: ignore
        column_name = column.slice.value.s  # type: ignore
    elif _is_ast_attribute(column):  # type: ignore
        column_name = column.attr  # type: ignore
    else:
        raise ValueError(
            "Expression type not supported. Formats supported: it.column_name or it['column_name']."
        )
    return col(column_name)


def ratio(expr: AstExpr):
    numerator = eval_spark_df(expr.args[0]) # type: ignore
    denominator = eval_spark_df(expr.args[1]) # type: ignore
    return numerator / denominator


def subtract(expr: AstExpr):
    e1 = eval_spark_df(expr.args[0]) # type: ignore
    e2 = eval_spark_df(expr.args[1]) # type: ignore
    return e1 / e2


def time_functions(dom_expr: AstExpr, spark_func):
    fmt = None
    column_name = dom_expr.args[0].attr
    if len(dom_expr.args) > 1:
        fmt = ast.literal_eval(dom_expr.args[1])
    return spark_func(to_timestamp(col(column_name), fmt))


def day_of_month(dom_expr: AstExpr):
    return time_functions(dom_expr, dayofmonth)


def day_of_week(dom_expr: AstExpr):
    return time_functions(dom_expr, dayofweek)


def day_of_year(dom_expr: AstExpr):
    return time_functions(dom_expr, dayofyear)


def hour(dom_expr: AstExpr):
    return time_functions(dom_expr, spark_hour)


def minute(dom_expr: AstExpr):
    return time_functions(dom_expr, spark_minute)


def month(dom_expr: AstExpr):
    return time_functions(dom_expr, spark_month)


def string_indexer(df, dom_expr: AstExpr):
    # XXX TODO XXX
    # column_name = dom_expr.args[0].attr
    # df = df.withColumnRenamed(
    #     column_name, "newColName"
    # )  # renaming because inputCol and outputCol can't be the same.
    # indexer = StringIndexer(inputCol="newColName", outputCol=new_column_name)
    # df = indexer.fit(df).transform(df)
    # df = df.drop("newColName")
    # return new_column_name, df
    pass
