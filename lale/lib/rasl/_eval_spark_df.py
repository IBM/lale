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

from lale.expressions import AstExpr, Expr, _it_column
from lale.helpers import _ast_func_id

try:
    import pyspark.sql.functions

    # noqa in the imports here because those get used dynamically and flake fails.
    from pyspark.sql.functions import col  # noqa
    from pyspark.sql.functions import isnan  # noqa
    from pyspark.sql.functions import lit  # noqa
    from pyspark.sql.functions import to_timestamp  # noqa
    from pyspark.sql.functions import hour as spark_hour  # noqa
    from pyspark.sql.functions import minute as spark_minute  # noqa
    from pyspark.sql.functions import month as spark_month  # noqa
    from pyspark.sql.types import LongType

    from pyspark.sql.functions import (  # noqa; isort: skip
        dayofmonth,
        dayofweek,
        dayofyear,
        floor as spark_floor,
        md5 as spark_md5,
        udf as spark_udf,
        when as spark_when,
    )

    spark_installed = True
except ImportError:
    spark_installed = False


def eval_expr_spark_df(expr: Expr):
    return _eval_ast_expr_spark_df(expr._expr)


def _eval_ast_expr_spark_df(expr: AstExpr):
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

    def visit_Attribute(self, node: ast.Attribute):
        column_name = _it_column(node)
        self.result = col(column_name)  # type: ignore

    def visit_Subscript(self, node: ast.Subscript):
        column_name = _it_column(node)
        self.result = col(column_name)  # type: ignore

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
            self.result = v1**v2
        elif isinstance(node.op, ast.BitAnd):
            self.result = v1 & v2
        elif isinstance(node.op, ast.BitOr):
            self.result = v1 | v2
        else:
            raise ValueError(f"""Unimplemented operator {ast.dump(node.op)}""")

    def visit_Compare(self, node: ast.Compare):
        self.visit(node.left)
        left = self.result
        assert len(node.ops) == len(node.comparators)
        if len(node.ops) != 1:  # need chained comparison in lale.expressions.Expr
            raise ValueError("Chained comparisons not supported yet.")
        self.visit(node.comparators[0])
        right = self.result
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            self.result = left == right  # type: ignore
        elif isinstance(op, ast.NotEq):
            self.result = left != right  # type: ignore
        elif isinstance(op, ast.Lt):
            self.result = left < right  # type: ignore
        elif isinstance(op, ast.LtE):
            self.result = left <= right  # type: ignore
        elif isinstance(op, ast.Gt):
            self.result = left > right  # type: ignore
        elif isinstance(op, ast.GtE):
            self.result = left >= right  # type: ignore
        else:
            raise ValueError(f"Unimplemented operator {ast.dump(op)}")

    def visit_Call(self, node: ast.Call):
        function_name = _ast_func_id(node.func)
        try:
            map_func_to_be_called = globals()[function_name]
        except KeyError:
            raise ValueError(f"""Unimplemented function {function_name}""")
        self.result = map_func_to_be_called(node)


def astype(call: ast.Call):
    dtype = ast.literal_eval(call.args[0])
    column = _eval_ast_expr_spark_df(call.args[1])  # type: ignore
    assert column is not None
    return column.astype(dtype)


def ite(call: ast.Call):
    cond = _eval_ast_expr_spark_df(call.args[0])  # type: ignore
    v1 = ast.literal_eval(call.args[1])
    v2 = ast.literal_eval(call.args[2])
    return spark_when(cond, v1).otherwise(v2)  # type: ignore


def hash(call: ast.Call):
    hashing_method = ast.literal_eval(call.args[0])
    column = _eval_ast_expr_spark_df(call.args[1])  # type: ignore
    if hashing_method == "md5":
        hash = spark_md5(column)  # type: ignore
    else:
        raise ValueError(f"Unimplementade hash function in Spark: {hashing_method}")
    return hash


def hash_mod(call: ast.Call):
    h_column = hash(call)
    N = ast.literal_eval(call.args[2])
    int16_mod_N = spark_udf((lambda x: int(x, 16) % N), LongType())
    return int16_mod_N(h_column)


def replace(call: ast.Call):
    column = _eval_ast_expr_spark_df(call.args[0])  # type: ignore
    mapping_dict = {}
    try:
        mapping_dict = ast.literal_eval(call.args[1].value)  # type: ignore
    except ValueError:
        mapping_dict_ast = call.args[1].value  # type: ignore
        # ast.literal_eval fails for `nan` with ValueError, we handle the case when
        # one of the keys is a `nan`. This is the case when using map with replace
        # in missing value imputation.
        for i, key in enumerate(mapping_dict_ast.keys):
            if key.id == "nan":
                mapping_dict["nan"] = ast.literal_eval(mapping_dict_ast.values[i])
            else:
                mapping_dict[
                    ast.literal_eval(ast.literal_eval(mapping_dict_ast.keys[i]))
                ] = ast.literal_eval(mapping_dict_ast.values[i])

    handle_unknown = ast.literal_eval(call.args[2])
    chain_of_whens = None
    for key, value in mapping_dict.items():
        if key == "nan":
            when_expr = isnan(column)  # type: ignore
        else:
            when_expr = column == key  # type: ignore
        if chain_of_whens is None:
            chain_of_whens = pyspark.sql.functions.when(when_expr, value)
        else:
            chain_of_whens = chain_of_whens.when(when_expr, value)
    if handle_unknown == "use_encoded_value":
        fallback = lit(ast.literal_eval(call.args[3]))
    else:
        fallback = column
    if chain_of_whens is None:
        result = fallback
    else:
        result = chain_of_whens.otherwise(fallback)
    return result


def identity(call: ast.Call):
    return _eval_ast_expr_spark_df(call.args[0])  # type: ignore


def time_functions(call, spark_func):
    column = _eval_ast_expr_spark_df(call.args[0])
    if len(call.args) > 1:
        fmt = ast.literal_eval(call.args[1])
        return spark_func(to_timestamp(column, format=fmt))  # type: ignore
    return spark_func(to_timestamp(column))  # type: ignore


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
