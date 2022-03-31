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

import numbers
import typing
from typing import Any, Tuple

import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators
from lale.expressions import count, it, median, mode, replace, sum
from lale.helpers import _is_df, _is_pandas_df, _is_spark_df, _is_spark_with_index
from lale.lib.dataframe import get_columns
from lale.lib.sklearn import simple_imputer
from lale.schemas import Enum

from .aggregate import Aggregate
from .map import Map
from .monoid import Monoid, MonoidableOperator


def _is_numeric_df(X):
    if _is_pandas_df(X):
        return X.shape[1] == X.select_dtypes(include=np.number).shape[1]
    elif _is_spark_df(X):
        from pyspark.sql.types import NumericType

        numeric_cols = [
            f.name for f in X.schema.fields if isinstance(f.dataType, NumericType)
        ]
        if _is_spark_with_index(X):
            for index_name in X.index_names:
                if index_name in numeric_cols:
                    numeric_cols.remove(index_name)
        return len(get_columns(X)) == len(numeric_cols)
    else:
        return False


def _is_string_df(X):
    if _is_pandas_df(X):
        return X.shape[1] == X.select_dtypes(include="object").shape[1]
    elif _is_spark_df(X):
        from pyspark.sql.types import StringType

        numeric_cols = [
            f.name for f in X.schema.fields if isinstance(f.dataType, StringType)
        ]
        return len(get_columns(X)) == len(numeric_cols)
    else:
        return False


class _SimpleImputerMonoid(Monoid):
    def __init__(self, *, feature_names_in_, lifted_statistics, strategy):
        self.feature_names_in_ = feature_names_in_
        self.lifted_statistics = lifted_statistics
        self.strategy = strategy

    def combine(self, other: "_SimpleImputerMonoid"):
        assert list(self.feature_names_in_) == list(other.feature_names_in_)
        if self.strategy == "constant":
            assert self.lifted_statistics.equals(other.lifted_statistics)
            combined_statistic = self.lifted_statistics
        elif self.strategy == "mean":
            combined_statistic = {}
            combined_statistic["sum"] = (
                self.lifted_statistics["sum"] + other.lifted_statistics["sum"]
            )
            combined_statistic["count"] = (
                self.lifted_statistics["count"] + other.lifted_statistics["count"]
            )
        else:
            raise ValueError(
                "_combine is only supported for imputation strategy `mean` and `constant`."
            )
        return _SimpleImputerMonoid(
            feature_names_in_=self.feature_names_in_,
            lifted_statistics=combined_statistic,
            strategy=self.strategy,
        )


class _SimpleImputerImpl(MonoidableOperator[_SimpleImputerMonoid]):
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        self._hyperparams = {}
        self._hyperparams["missing_values"] = missing_values
        self._hyperparams["strategy"] = strategy
        self._hyperparams["fill_value"] = fill_value
        self._hyperparams["verbose"] = verbose
        if not copy:
            raise ValueError("This implementation only supports `copy=True`.")
        self._hyperparams["copy"] = copy
        if add_indicator:
            raise ValueError("This implementation only supports `add_indicator=False`.")
        self._hyperparams["add_indicator"] = add_indicator
        # the `indicator_`` property is always None as we do not support `add_indicator=True`
        self.indicator_ = None

    def to_monoid(self, v: Tuple[Any, Any]):
        hyperparams = self._hyperparams
        X, _ = v
        feature_names_in_ = get_columns(X)
        agg_data = None
        # learn the values to be imputed
        strategy = hyperparams["strategy"]
        if strategy == "constant":
            fill_value = _SimpleImputerImpl._get_fill_value(X, hyperparams)
            agg_data = [[fill_value for col in get_columns(X)]]
            lifted_statistics = pd.DataFrame(agg_data, columns=get_columns(X))
        elif strategy == "mean":
            agg_op_sum = Aggregate(
                columns={c: sum(it[c]) for c in get_columns(X)},
                exclude_value=hyperparams["missing_values"],
            )
            agg_op_count = Aggregate(
                columns={c: count(it[c]) for c in get_columns(X)},
                exclude_value=hyperparams["missing_values"],
            )
            lifted_statistics = {}
            agg_sum = agg_op_sum.transform(X)
            if agg_sum is not None and _is_spark_df(agg_sum):
                agg_sum = agg_sum.toPandas()
            agg_count = agg_op_count.transform(X)
            if agg_count is not None and _is_spark_df(agg_count):
                agg_count = agg_count.toPandas()
            lifted_statistics["sum"] = agg_sum
            lifted_statistics["count"] = agg_count
        else:
            raise ValueError(
                "SimpleImputer can create a Monoind only for imputation strategy `mean` and `constant`."
            )
        return _SimpleImputerMonoid(
            feature_names_in_=feature_names_in_,
            lifted_statistics=lifted_statistics,
            strategy=strategy,
        )

    def from_monoid(self, lifted):
        self._monoid = lifted
        self.feature_names_in_ = lifted.feature_names_in_
        self.n_features_in_ = len(self.feature_names_in_)
        _lifted_statistics = lifted.lifted_statistics
        strategy = self._hyperparams["strategy"]
        if strategy == "constant":
            self.statistics_ = _lifted_statistics.to_numpy()[0]
        elif strategy == "mean":
            self.statistics_ = (
                _lifted_statistics["sum"] / _lifted_statistics["count"]
            ).to_numpy()[0]
        self._transformer = None

    def fit(self, X, y=None):

        self._validate_input(X)

        agg_op = None
        agg_data = None
        # learn the values to be imputed
        if self._hyperparams["strategy"] in ["mean", "constant"]:
            lifted = self.to_monoid((X, y))
            self.from_monoid(lifted)
            return self
        elif self._hyperparams["strategy"] == "median":
            agg_op = Aggregate(
                columns={c: median(it[c]) for c in get_columns(X)},
                exclude_value=self._hyperparams["missing_values"],
            )
        elif self._hyperparams["strategy"] == "most_frequent":
            agg_op = Aggregate(
                columns={c: mode(it[c]) for c in get_columns(X)},
                exclude_value=self._hyperparams["missing_values"],
            )
        if agg_data is None and agg_op is not None:
            agg_data = agg_op.transform(X)
        self.feature_names_in_ = get_columns(X)
        self.n_features_in_ = len(self.feature_names_in_)
        if agg_data is not None and _is_spark_df(agg_data):
            agg_data = agg_data.toPandas()
        if agg_data is not None and _is_pandas_df(agg_data):
            self.statistics_ = agg_data.to_numpy()[
                0
            ]  # Converting from a 2-d array to 1-d
        self._transformer = None
        return self

    def _build_transformer(self):
        # prepare the transformer
        transformer = Map(
            columns={
                col_name: replace(
                    it[col_name],
                    {self._hyperparams["missing_values"]: self.statistics_[col_idx]},
                )
                for col_idx, col_name in enumerate(self.feature_names_in_)
            }
        )
        return transformer

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    @staticmethod
    def _get_fill_value(X, hyperparams):
        # assign appropriate value to fill_value depending on the datatype.
        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if hyperparams["fill_value"] is None:
            if _is_numeric_df(X):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = hyperparams["fill_value"]
        # validate that fill_value is numerical for numerical data
        if (
            hyperparams["strategy"] == "constant"
            and _is_numeric_df(X)
            and not isinstance(fill_value, numbers.Real)
        ):
            raise ValueError(
                "'fill_value'={0} is invalid. Expected a "
                "numerical value when imputing numerical "
                "data".format(fill_value)
            )
        return fill_value

    def _validate_input(self, X):
        # validate that the dataset is either a pandas dataframe or spark.
        # For example, sparse matrix is not allowed.
        if not _is_df(X):
            raise ValueError(
                f"""Unsupported type(X) {type(X)} for SimpleImputer.
            Only pandas.DataFrame or pyspark.sql.DataFrame are allowed."""
            )
        # validate input to check the correct dtype and strategy
        # `mean` and `median` are not applicable to string inputs
        if not _is_numeric_df(X) and self._hyperparams["strategy"] in [
            "mean",
            "median",
        ]:
            raise ValueError(
                "Cannot use {} strategy with non-numeric data.".format(
                    self._hyperparams["strategy"]
                )
            )

        # Check that missing_values are the right type
        if _is_numeric_df(X) and not isinstance(
            self._hyperparams["missing_values"], numbers.Real
        ):
            raise ValueError(
                "'X' and 'missing_values' types are expected to be"
                " both numerical. Got X.dtypes={} and "
                " type(missing_values)={}.".format(
                    X.dtypes, type(self._hyperparams["missing_values"])
                )
            )


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Relational algebra reimplementation of scikit-learn's `SimpleImputer`_.
Works on both pandas and Spark dataframes by using `Aggregate`_ for `fit` and `Map`_ for `transform`, which in turn use the appropriate backend.

.. _`SimpleImputer`: https://scikit-learn.org/stable/modules/generated/sklearn.imputer.SimpleImputer.html
.. _`Aggregate`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.aggregate.html
.. _`Map`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.simple_imputer.html",
    "type": "object",
    "tags": {
        "pre": [],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": simple_imputer._hyperparams_schema,
        "input_fit": simple_imputer._input_fit_schema,
        "input_transform": simple_imputer._input_transform_schema,
        "output_transform": simple_imputer._output_transform_schema,
    },
}

SimpleImputer = lale.operators.make_operator(_SimpleImputerImpl, _combined_schemas)

SimpleImputer = typing.cast(
    lale.operators.PlannedIndividualOp,
    SimpleImputer.customize_schema(
        copy=Enum(
            values=[True],
            desc="`copy=True` is the only value currently supported by this implementation",
            default=True,
        ),
        add_indicator=Enum(
            values=[False],
            desc="`add_indicator=False` is the only value currently supported by this implementation",
            default=False,
        ),
    ),
)

lale.docstrings.set_docstrings(SimpleImputer)
