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

import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators
from lale.expressions import it, mean, median, mode, replace
from lale.helpers import _is_df, _is_pandas_df, _is_spark_df
from lale.lib.sklearn import simple_imputer
from lale.schemas import Enum

from .aggregate import Aggregate
from .map import Map


def _is_numeric_df(X):
    if _is_pandas_df(X):
        return X.shape[1] == X.select_dtypes(include=np.number).shape[1]
    elif _is_spark_df(X):
        from pyspark.sql.types import NumericType

        numeric_cols = [
            f.name for f in X.schema.fields if isinstance(f.dataType, NumericType)
        ]
        return len(X.columns) == len(numeric_cols)
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
        return len(X.columns) == len(numeric_cols)
    else:
        return False


class _SimpleImputerImpl:
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):

        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        if not copy:
            raise ValueError("This implementation only supports `copy=True`.")

        if add_indicator:
            raise ValueError("This implementation only supports `add_indicator=False`.")
        # the `indicator_`` property is always None as we do not support `add_indictor=True`
        self.indicator_ = None

    def fit(self, X, y=None):

        self._validate_input(X)

        # assign appropriate value to fill_value depending on the datatype.
        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if _is_numeric_df(X):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        # validate that fill_value is numerical for numerical data
        if (
            self.strategy == "constant"
            and _is_numeric_df(X)
            and not isinstance(fill_value, numbers.Real)
        ):
            raise ValueError(
                "'fill_value'={0} is invalid. Expected a "
                "numerical value when imputing numerical "
                "data".format(fill_value)
            )

        # set attribute values
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = X.columns

        agg_op = None
        agg_data = None
        # learn the values to be imputed
        if self.strategy == "mean":
            agg_op = Aggregate(
                columns={c: mean(it[c]) for c in X.columns},
                exclude_value=self.missing_values,
            )
        elif self.strategy == "median":
            agg_op = Aggregate(
                columns={c: median(it[c]) for c in X.columns},
                exclude_value=self.missing_values,
            )
        elif self.strategy == "most_frequent":
            agg_op = Aggregate(
                columns={c: mode(it[c]) for c in X.columns},
                exclude_value=self.missing_values,
            )
        elif self.strategy == "constant":
            agg_data = [[fill_value for col in X.columns]]
            agg_data = pd.DataFrame(agg_data, columns=X.columns)
        if agg_data is None and agg_op is not None:
            agg_data = agg_op.transform(X)
        if agg_data is not None and _is_spark_df(agg_data):
            agg_data = agg_data.toPandas()

        if agg_data is not None and _is_pandas_df(agg_data):
            self.statistics_ = agg_data.to_numpy()[
                0
            ]  # Converting from a 2-d array to 1-d
        # prepare the transformer
        if agg_data is not None:
            self.transformer = Map(
                columns={
                    col_name: replace(
                        it[col_name], {self.missing_values: agg_data.iloc[0, col_idx]}
                    )
                    for col_idx, col_name in enumerate(X.columns)
                }
            )
        return self

    def transform(self, X):
        return self.transformer.transform(X)

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
        if not _is_numeric_df(X) and self.strategy in ["mean", "median"]:
            raise ValueError(
                "Cannot use {} strategy with non-numeric data.".format(self.strategy)
            )

        # Check that missing_values are the right type
        if _is_numeric_df(X) and not isinstance(self.missing_values, numbers.Real):
            raise ValueError(
                "'X' and 'missing_values' types are expected to be"
                " both numerical. Got X.dtypes={} and "
                " type(missing_values)={}.".format(X.dtypes, type(self.missing_values))
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
