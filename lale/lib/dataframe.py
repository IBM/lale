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

"""
Common interface to manipulate different type of dataframes supported in Lale.
"""

from typing import List, Union

import numpy as np
import pandas as pd

from lale.helpers import (
    _is_pandas_df,
    _is_pandas_series,
    _is_spark_df,
    _is_spark_with_index,
)

column_index = Union[str, int]


def get_columns(df) -> List[column_index]:
    if _is_pandas_series(df):
        return pd.Series([df.name])
    if _is_pandas_df(df):
        return df.columns
    if _is_spark_with_index(df):
        return pd.Series(df.columns_without_indexes)
    if _is_spark_df(df):
        return df.columns
    if isinstance(df, np.ndarray):
        # should have more asserts here
        _, num_cols = df.shape
        return list(range(num_cols))
    assert False, type(df)


def select_col(df, col: column_index):
    if isinstance(df, np.ndarray):
        return df[:, col]  # type: ignore
    elif _is_pandas_df(df):
        return df[col]
    elif _is_spark_df(df):
        return df.select(col)
    else:
        raise ValueError(f"Unsupported series type {type(df)}")


def count(df):
    if isinstance(df, np.ndarray):
        return df.size
    if _is_pandas_df(df) or _is_pandas_series(df):
        return len(df)
    elif _is_spark_df(df):
        return df.count()
    else:
        return len(df)


def make_series_distinct(df):
    if isinstance(df, np.ndarray):
        return np.unique(df)
    elif isinstance(df, pd.Series):
        return df.unique()
    elif _is_spark_df(df):
        return df.distinct()
    else:
        raise ValueError(f"Unsupported series type {type(df)}")


def make_series_concat(df1, df2):
    if isinstance(df1, np.ndarray):
        assert isinstance(df2, np.ndarray)
        return np.concatenate((df1, df2))
    elif isinstance(df1, pd.Series):
        assert isinstance(df2, pd.Series)
        return pd.concat([df1, df2])
    elif _is_spark_df(df1):
        assert _is_spark_df(df2)
        return df1.union(df2)
    else:
        raise ValueError(f"Unsupported series type {type(df1)}")
