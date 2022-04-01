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
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import numpy as np
from typing_extensions import Protocol

from ..dataframe import (
    column_index,
    count,
    get_columns,
    make_series_concat,
    make_series_distinct,
    select_col,
)

try:
    import pyspark.sql.functions

    spark_installed = True
except ImportError:
    spark_installed = False


from lale.helpers import _is_pandas_df, _is_spark_df

from .monoid import Monoid, MonoidFactory


class _column_distinct_count_data(Monoid):
    def __init__(self, df, limit: Optional[int] = None):
        self.limit = limit
        self.df = make_series_distinct(df)

    def __len__(self):
        return count(self.df)

    @property
    def is_absorbing(self):
        if self.limit is None:
            return False
        else:
            return len(self) > self.limit

    def combine(self, other: "_column_distinct_count_data"):
        if self.is_absorbing:
            return self
        elif other.is_absorbing:
            return other
        else:
            c = make_series_concat(self.df, other.df)
            return _column_distinct_count_data(c, limit=self.limit)


# numpy or sparkdf or pandas
_Batch = Any


class count_distinct_column(MonoidFactory[_Batch, int, _column_distinct_count_data]):
    """
    Counts the number of distinct elements in a given column.  If a limit is specified,
    then, once the limit is reached, the count may no longer be accurate
    (but will always remain over the limit).
    """

    def __init__(self, col: column_index, limit: Optional[int] = None):
        self._col = col
        self._limit = limit

    def to_monoid(self, df) -> _column_distinct_count_data:
        c = select_col(df, self._col)
        return _column_distinct_count_data(c, limit=self._limit)

    def from_monoid(self, v: _column_distinct_count_data) -> int:
        return len(v)


class categorical_column(MonoidFactory[_Batch, bool, _column_distinct_count_data]):
    """
    Determines if a column should be considered categorical,
    by seeing if there are more than threshold distinct values in it
    """

    def __init__(self, col: column_index, threshold: int = 5):
        self._col = col
        self._threshold = threshold

    def to_monoid(self, df) -> _column_distinct_count_data:
        c = select_col(df, self._col)
        return _column_distinct_count_data(c, limit=self._threshold)

    def from_monoid(self, v: _column_distinct_count_data) -> bool:
        return not v.is_absorbing


class make_categorical_column:
    def __init__(self, threshold=5):
        self._threshold = threshold

    def __call__(self, col):
        return categorical_column(col, threshold=self._threshold)


_D = TypeVar("_D", bound=Monoid)


class DictMonoid(Generic[_D], Monoid):
    """
    Given a monoid, this class lifts it to a dictionary pointwise
    """

    def __init__(self, m: Dict[Any, _D]):
        self._m = m

    def combine(self, other: "DictMonoid[_D]"):
        r = {k: self._m[k].combine(other._m[k]) for k in self._m.keys()}
        return DictMonoid(r)

    @property
    def is_absorbing(self):
        return all(v.is_absorbing for v in self._m.values())


class ColumnSelector(MonoidFactory[_Batch, List[column_index], _D], Protocol):
    def __call__(self, df) -> List[column_index]:
        return self.from_monoid(self.to_monoid(df))


class ColumnMonoidFactory(ColumnSelector[DictMonoid[_D]]):
    """
    Given a MonoidFactory for deciding if a given column is valid,
    This returns the list of valid columns
    """

    _makers: Optional[Dict[column_index, MonoidFactory[_Batch, bool, _D]]]

    def __init__(
        self, col_maker: Callable[[column_index], MonoidFactory[_Batch, bool, _D]]
    ):
        self._col_maker = col_maker
        self._makers = None

    def _get_makers(self, df):
        makers = self._makers
        if makers is None:
            indices = get_columns(df)
            makers = {k: self._col_maker(k) for k in indices}
            self._makers = makers
        return makers

    def to_monoid(self, df):
        makers = self._get_makers(df)
        return DictMonoid({k: v.to_monoid(df) for k, v in makers.items()})

    def from_monoid(self, d: DictMonoid[_D]) -> List[column_index]:
        makers = self._makers
        assert makers is not None
        return [k for k, v in makers.items() if v.from_monoid(d._m[k])]


class categorical(ColumnMonoidFactory):
    """Creates a MonoidFactory (and callable) for projecting categorical columns with sklearn's ColumnTransformer or Lale's Project operator.

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
        super().__init__(make_categorical_column(max_values))


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
