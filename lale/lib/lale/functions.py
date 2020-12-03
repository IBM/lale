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
import datetime
from itertools import chain
from typing import Any, Union

import numpy as np
import pandas as pd
from pyspark.sql.dataframe import DataFrame as spark_df
from pyspark.sql.functions import create_map, lit

from lale.expressions import Expr


class categorical:
    """Creates a callable for projecting categorical columns with sklearn's ColumnTransformer or Lale's Project operator.

    Parameters
    ----------
    max_values : int

        Maximum number of unique values in a column for it to be considered categorical.

    Returns
    -------
    callable
        Function that, given a dataset X, returns a list of columns,
        containing either string column names or integer column indices."""

    def __init__(self, max_values: int = 5):
        self._max_values = max_values

    def __repr__(self):
        return f"lale.lib.lale.categorical(max_values={self._max_values})"

    def __call__(self, X):
        def is_categorical(column_values):
            unique_values = set()
            for val in column_values:
                if val not in unique_values:
                    unique_values.add(val)
                    if len(unique_values) > self._max_values:
                        return False
            return True

        if isinstance(X, pd.DataFrame):
            result = [c for c in X.columns if is_categorical(X[c])]
        elif isinstance(X, np.ndarray):
            result = [c for c in range(X.shape[1]) if is_categorical(X[:, c])]
        else:
            raise TypeError(f"unexpected type {type(X)}")
        return result


class date_time:
    """Creates a callable for projecting date/time columns with sklearn's ColumnTransformer or Lale's Project operator.

    Parameters
    ----------
    fmt : str

        Format string for `strptime()`, see https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

    Returns
    -------
    callable
        Function that, given a dataset X, returns a list of columns,
        containing either string column names or integer column indices."""

    def __init__(self, fmt):
        self._fmt = fmt

    def __repr__(self):
        return f"lale.lib.lale.date_time(fmt={self._fmt})"

    def __call__(self, X):
        def is_date_time(column_values):
            try:
                for val in column_values:
                    if isinstance(val, str):
                        datetime.datetime.strptime(val, self._fmt)
                    else:
                        return False
            except ValueError:
                return False
            return True

        if isinstance(X, pd.DataFrame):
            result = [c for c in X.columns if is_date_time(X[c])]
        elif isinstance(X, np.ndarray):
            result = [c for c in range(X.shape[1]) if is_date_time(X[:, c])]
        else:
            raise TypeError(f"unexpected type {type(X)}")
        return result


def replace(
    df: Union[pd.DataFrame, spark_df], replace_expr: Expr, new_column_name: str
):
    re: Any = replace_expr._expr
    column_name = re.args[0].attr
    if new_column_name is None:
        new_column_name = column_name
    mapping_dict = ast.literal_eval(re.args[1].value)
    if isinstance(df, pd.DataFrame):
        new_column = df[column_name].replace(mapping_dict)
        df[new_column_name] = new_column
        if new_column_name != column_name:
            del df[column_name]
    elif isinstance(df, spark_df):
        mapping_expr = create_map([lit(x) for x in chain(*mapping_dict.items())])
        df = df.withColumn(new_column_name, mapping_expr[df[column_name]])
        if new_column_name != column_name:
            df = df.drop(column_name)
    else:
        raise ValueError(
            "function replace supports only Pandas dataframes or spark dataframes."
        )
    return new_column_name, df


def day_of_month(df: pd.DataFrame, dom_expr: Expr):
    fmt = None
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    if len(de.args) > 1:
        fmt = ast.literal_eval(de.args[1])
    df[column_name] = pd.to_datetime(df[column_name], format=fmt)
    return column_name, df[column_name].dt.day


def day_of_week(df: pd.DataFrame, dom_expr: Expr):
    fmt = None
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    if len(de.args) > 1:
        fmt = ast.literal_eval(de.args[1])
    df[column_name] = pd.to_datetime(df[column_name], format=fmt)
    return column_name, df[column_name].dt.weekday


def day_of_year(df: pd.DataFrame, dom_expr: Expr):
    fmt = None
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    if len(de.args) > 1:
        fmt = ast.literal_eval(de.args[1])
    df[column_name] = pd.to_datetime(df[column_name], format=fmt)
    return column_name, df[column_name].dt.dayofyear


def hour(df: pd.DataFrame, dom_expr: Expr):
    fmt = None
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    if len(de.args) > 1:
        fmt = ast.literal_eval(de.args[1])
    df[column_name] = pd.to_datetime(df[column_name], format=fmt)
    return column_name, df[column_name].dt.hour


def minute(df: pd.DataFrame, dom_expr: Expr):
    fmt = None
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    if len(de.args) > 1:
        fmt = ast.literal_eval(de.args[1])
    df[column_name] = pd.to_datetime(df[column_name], format=fmt)
    return column_name, df[column_name].dt.minute


def month(df: pd.DataFrame, dom_expr: Expr):
    fmt = None
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    if len(de.args) > 1:
        fmt = ast.literal_eval(de.args[1])
    df[column_name] = pd.to_datetime(df[column_name], format=fmt)
    return column_name, df[column_name].dt.month


def string_indexer(df: pd.DataFrame, dom_expr: Expr):
    de: Any = dom_expr._expr
    column_name = de.args[0].attr
    sorted_indices = df[column_name].value_counts().index
    return (
        column_name,
        df[column_name].map(
            dict(zip(sorted_indices, range(1, len(sorted_indices) + 1)))
        ),
    )
