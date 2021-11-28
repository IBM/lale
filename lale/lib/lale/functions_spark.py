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
from itertools import chain

from lale.helpers import eval_spark_df

from lale.expressions import AstExpr
from lale.helpers import (
    _is_ast_attribute,
    _is_ast_subscript,
)

try:
    import pyspark.sql.functions as pysql
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


# # functions for filter
# def filter_isnan(column_name: str):
#     return df[df[column_name].isnull()]


# def filter_isnotnan(column_name: str):
#     return df[df[column_name].notnull()]


# def filter_isnull(column_name: str):
#     return df[df[column_name].isnull()]


# def filter_isnotnull(column_name: str):
#     return df[df[column_name].notnull()]
