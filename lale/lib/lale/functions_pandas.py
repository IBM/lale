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
from typing import Any

import numpy as np
import pandas as pd

from lale.helpers import pandas_df_eval

from lale.expressions import AstExpr
from lale.helpers import (
    _is_ast_attribute,
    _is_ast_subscript,
)


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
    numerator = pandas_df_eval(df, expr.args[0]) # type: ignore
    denominator = pandas_df_eval(df, expr.args[1]) # type: ignore
    return numerator / denominator


def subtract(df: Any, expr: AstExpr):
    e1 = pandas_df_eval(df, expr.args[0]) # type: ignore
    e2 = pandas_df_eval(df, expr.args[1]) # type: ignore
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


# functions for aggregate
def grouped_sum():
    return pysql.sum


def grouped_max():
    return pysql.max


def grouped_min():
    return pysql.min


def grouped_count():
    return pysql.count


def grouped_mean():
    return pysql.mean


def grouped_first():
    return pysql.first


# functions for filter
def filter_isnan(df: Any, column_name: str):
    return df[df[column_name].isnull()]


def filter_isnotnan(df: Any, column_name: str):
    return df[df[column_name].notnull()]


def filter_isnull(df: Any, column_name: str):
    return df[df[column_name].isnull()]


def filter_isnotnull(df: Any, column_name: str):
    return df[df[column_name].notnull()]
