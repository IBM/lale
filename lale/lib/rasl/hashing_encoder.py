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

from functools import reduce
from typing import Any, List, Optional, Tuple

import pandas as pd

import lale.docstrings
import lale.helpers
import lale.operators
from lale.expressions import Expr, hash_mod, it, ite
from lale.helpers import _is_pandas_df, _is_spark_df
from lale.lib.dataframe import count, get_columns
from lale.lib.sklearn import hashing_encoder

from .map import Map
from .monoid import Monoid, MonoidableOperator


# Based on https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/utils.py
def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    if _is_pandas_df(df):
        for idx, dt in enumerate(df.dtypes):
            if dt == "object" or is_category(dt):
                obj_cols.append(df.columns.values[idx])
    elif _is_spark_df(df):
        for idx, (col, dt) in enumerate(df.dtypes):
            if dt == "string":
                obj_cols.append(col)
    else:
        assert False

    return obj_cols


# From https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/utils.py
def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


class _HashingEncoderMonoid(Monoid):
    def __init__(self, *, n_samples_seen_, feature_names):
        self.n_samples_seen_ = n_samples_seen_
        self.feature_names = feature_names

    def combine(self, other: "_HashingEncoderMonoid"):
        assert list(self.feature_names) == list(other.feature_names)
        n_samples_seen_ = self.n_samples_seen_ + other.n_samples_seen_
        return _HashingEncoderMonoid(
            n_samples_seen_=n_samples_seen_, feature_names=self.feature_names
        )


class _HashingEncoderImpl(MonoidableOperator[_HashingEncoderMonoid]):
    def __init__(
        self,
        *,
        n_components=8,
        cols: Optional[List[str]] = None,
        # drop_invariant=False,
        # return_df=True,
        hash_method="md5",
    ):
        self._hyperparams = {
            "n_components": n_components,
            "cols": cols,
            # "drop_invariant": drop_invariant,
            "hash_method": hash_method,
        }
        self._dim = None

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer(X)
        return self._transformer.transform(X)

    @property
    def n_samples_seen_(self):
        return getattr(self._monoid, "n_samples_seen_", 0)

    @property
    def feature_names(self):
        return getattr(self._monoid, "feature_names", None)

    def from_monoid(self, lifted):
        self._monoid = lifted
        self._transformer = None

    def _build_transformer(self, X):
        cols = self._hyperparams["cols"]
        hash_method = self._hyperparams["hash_method"]
        N = self._hyperparams["n_components"]
        columns_hash = {
            col_name: hash_mod(hash_method, it[col_name], N) for col_name in cols
        }
        columns_cat = {
            f"col_{i}": reduce(
                Expr.__add__,
                [ite(it[col_name] == i, 1, 0) for col_name in cols],
            )
            for i in range(N)
        }
        hash = Map(columns=columns_hash, remainder="passthrough")
        encode = Map(columns=columns_cat, remainder="passthrough")
        return hash >> encode

    def to_monoid(self, v: Tuple[Any, Any]):
        X, y = v
        cols = self._hyperparams["cols"]
        if cols is None:
            cols = get_obj_cols(X)
            self._hyperparams["cols"] = cols

        N = self._hyperparams["n_components"]
        feature_names_cat = [f"col_{i}" for i in range(N)]
        feature_names_num = [col for col in get_columns(X) if col not in cols]
        feature_names = feature_names_cat + feature_names_num  # type: ignore
        n_samples_seen_ = count(X)
        return _HashingEncoderMonoid(
            n_samples_seen_=n_samples_seen_, feature_names=feature_names
        )

    # https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/hashing.py
    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.
        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not isinstance(self.feature_names, list):
            raise ValueError(
                "Must fit data first. Affected feature names are not known before."
            )
        else:
            return self.feature_names


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Relational algebra reimplementation of scikit-learn contrib's `HashingEncoder`_ transformer.
Works on both pandas and Spark dataframes by using `Map`_ for `transform`, which in turn use the appropriate backend.

.. _`HashingEncoder`: https://contrib.scikit-learn.org/category_encoders/hashing.html
.. _`Map`: https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.map.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.hashing_encoder.html",
    "type": "object",
    "tags": {
        "pre": ["categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": hashing_encoder._hyperparams_schema,
        "input_fit": hashing_encoder._input_fit_schema,
        "input_transform": hashing_encoder._input_transform_schema,
        "output_transform": hashing_encoder._output_transform_schema,
    },
}

HashingEncoder = lale.operators.make_operator(_HashingEncoderImpl, _combined_schemas)

# HashingEncoder = typing.cast(
#     lale.operators.PlannedIndividualOp,
# )

lale.docstrings.set_docstrings(HashingEncoder)
