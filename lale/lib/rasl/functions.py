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

import datetime
from typing import Any

import numpy as np

try:
    import pyspark.sql.functions

    spark_installed = True
except ImportError:
    spark_installed = False


from lale.helpers import _is_pandas_df, _is_spark_df


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
        return f"lale.lib.rasl.categorical(max_values={self._max_values})"

    def __call__(self, X):
        def is_categorical(column_values):
            unique_values = set()
            for val in column_values:
                if val not in unique_values:
                    unique_values.add(val)
                    if len(unique_values) > self._max_values:
                        return False
            return True

        if _is_pandas_df(X):
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
        return f"lale.lib.rasl.date_time(fmt={self._fmt})"

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

        if _is_pandas_df(X):
            result = [c for c in X.columns if is_date_time(X[c])]
        elif isinstance(X, np.ndarray):
            result = [c for c in range(X.shape[1]) if is_date_time(X[:, c])]
        else:
            raise TypeError(f"unexpected type {type(X)}")
        return result


# functions for filter
def filter_isnan(df: Any, column_name: str):
    if _is_pandas_df(df):
        return df[df[column_name].isnull()]
    elif spark_installed and _is_spark_df(df):
        return df.filter(pyspark.sql.functions.isnan(df[column_name]))
    else:
        raise ValueError(
            "the filter isnan supports only Pandas dataframes or spark dataframes."
        )


def filter_isnotnan(df: Any, column_name: str):
    if _is_pandas_df(df):
        return df[df[column_name].notnull()]
    elif spark_installed and _is_spark_df(df):
        return df.filter(~pyspark.sql.functions.isnan(df[column_name]))
    else:
        raise ValueError(
            "the filter isnotnan supports only Pandas dataframes or spark dataframes."
        )


def filter_isnull(df: Any, column_name: str):
    if _is_pandas_df(df):
        return df[df[column_name].isnull()]
    elif spark_installed and _is_spark_df(df):
        return df.filter(pyspark.sql.functions.isnull(df[column_name]))
    else:
        raise ValueError(
            "the filter isnan supports only Pandas dataframes or spark dataframes."
        )


def filter_isnotnull(df: Any, column_name: str):
    if _is_pandas_df(df):
        return df[df[column_name].notnull()]
    elif spark_installed and _is_spark_df(df):
        return df.filter(~pyspark.sql.functions.isnull(df[column_name]))
    else:
        raise ValueError(
            "the filter isnotnan supports only Pandas dataframes or spark dataframes."
        )
