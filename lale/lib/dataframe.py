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

from typing import List

import pandas as pd

from lale.helpers import (
    _is_pandas_df,
    _is_pandas_series,
    _is_spark_df,
    _is_spark_with_index,
)


def get_columns(df) -> List[str]:
    if _is_pandas_series(df):
        return pd.Series([df.name])
    if _is_pandas_df(df):
        return df.columns
    if _is_spark_with_index(df):
        return pd.Series(df.columns_without_index)
    if _is_spark_df(df):
        return df.columns
    assert False, type(df)


def count(df):
    if _is_pandas_df(df) or _is_pandas_series(df):
        return len(df)
    elif _is_spark_df(df):
        return df.count()
