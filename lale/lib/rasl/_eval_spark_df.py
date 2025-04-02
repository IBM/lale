# Copyright 2021-2022 IBM Corporation
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
    # noqa in the imports here because those get used dynamically and flake fails.
    from pyspark.sql.functions import col
    from pyspark.sql.functions import hour as spark_hour
    from pyspark.sql.functions import isnan, isnull, lit
    from pyspark.sql.functions import minute as spark_minute
    from pyspark.sql.functions import month as spark_month
    from pyspark.sql.functions import to_timestamp
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
    lit = None
    col = None
    to_timestamp = None
    isnan = None
    isnull = None
    spark_udf = None
    spark_floor = None
    LongType = None
    spark_when = None
    dayofmonth = None
    dayofweek = None
    dayofyear = None
    spark_floor = None
    spark_md5 = None
    spark_udf = None
    spark_when = None
    spark_hour = None
    spark_minute = None
    spark_month = None

    spark_installed = False


def eval_expr_spark_df(expr: Expr):
    return _eval_ast_expr_spark_df(expr.expr)


def _eval_ast_expr_spark_df(expr: AstExpr):
    evaluator = _SparkEvaluator()
    evaluator.visit(expr)
    return evaluator.result


class _SparkEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.result = None

    def visit_Constant(self, node: ast.Constant):
        assert lit is not None
        self.result = lit(node.value)

    def visit_Attribute(self, node: ast.Attribute):
        column_name = _it_column(node)
        assert col is not None
        self.result = col(column_name)  # type: ignore

    def visit_Subscript(self, node: ast.Subscript):
        column_name = _it_column(node)
        assert col is not None
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
            assert spark_floor is not None
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
        except KeyError as exc:
            raise ValueError(f"""Unimplemented function {function_name}""") from exc
        self.result = map_func_to_be_called(node)


def astype(call: ast.Call):
    dtype = ast.literal_eval(call.args[0])
    column = _eval_ast_expr_spark_df(call.args[1])  # type: ignore
    assert column is not None
    return column.astype(dtype)


def ite(call: ast.Call):
    cond = _eval_ast_expr_spark_df(call.args[0])  # type: ignore
    v1 = _eval_ast_expr_spark_df(call.args[1])  # type: ignore
    v2 = _eval_ast_expr_spark_df(call.args[2])  # type: ignore
    return spark_when(cond, v1).otherwise(v2)  # type: ignore


def hash(call: ast.Call):  # pylint:disable=redefined-builtin
    hashing_method = ast.literal_eval(call.args[0])
    column = _eval_ast_expr_spark_df(call.args[1])  # type: ignore
    if hashing_method == "md5":
        assert spark_md5 is not None
        assert column is not None
        hash_fun = spark_md5(column)
    else:
        raise ValueError(f"Unimplementade hash function in Spark: {hashing_method}")
    return hash_fun


def hash_mod(call: ast.Call):
    h_column = hash(call)
    N = ast.literal_eval(call.args[2])
    assert spark_udf is not None
    assert LongType is not None
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
            if hasattr(key, "id") and key.id == "nan":
                mapping_dict["nan"] = ast.literal_eval(mapping_dict_ast.values[i])
            else:
                mapping_dict[ast.literal_eval(mapping_dict_ast.keys[i])] = (
                    ast.literal_eval(mapping_dict_ast.values[i])
                )

    handle_unknown = ast.literal_eval(call.args[2])
    chain_of_whens = None
    assert column is not None
    for key, value in mapping_dict.items():
        if key == "nan":
            assert isnan is not None
            when_expr = isnan(column)
        elif key is None:
            assert isnull is not None
            when_expr = isnull(column)
        else:
            when_expr = column == key
        if chain_of_whens is None:
            assert spark_when is not None
            chain_of_whens = spark_when(when_expr, value)
        else:
            chain_of_whens = chain_of_whens.when(when_expr, value)
    if handle_unknown == "use_encoded_value":
        assert lit is not None
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
    assert to_timestamp is not None
    assert column is not None
    if len(call.args) > 1:
        fmt = ast.literal_eval(call.args[1])
        return spark_func(to_timestamp(column, format=fmt))
    return spark_func(to_timestamp(column))


def day_of_month(call: ast.Call):
    assert dayofmonth is not None
    return time_functions(call, dayofmonth)


def day_of_week(call: ast.Call):
    assert dayofweek is not None
    return time_functions(call, dayofweek)


def day_of_year(call: ast.Call):
    assert dayofyear is not None
    return time_functions(call, dayofyear)


def hour(call: ast.Call):
    assert spark_hour is not None
    return time_functions(call, spark_hour)


def minute(call: ast.Call):
    assert spark_minute is not None
    return time_functions(call, spark_minute)


def month(call: ast.Call):
    assert spark_month is not None
    return time_functions(call, spark_month)
