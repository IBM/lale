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

import pandas as pd

import lale.docstrings
import lale.helpers
import lale.operators
from lale.expressions import Expr, hash, it, ite
from lale.helpers import _is_pandas_df, _is_spark_df
from lale.lib.dataframe import count, get_columns
from lale.lib.sklearn import hashing_encoder

from .map import Map


# From https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/utils.py
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
        assert False, "Not yet implemented"

    return obj_cols


# From https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/utils.py
def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


class _HashingEncoderImpl:
    def __init__(
        self,
        *,
        n_components=8,
        cols=None,
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
        self.n_samples_seen_ = 0
        self._dim = None
        self.feature_names = None

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, self._hyperparams))
        return self

    def partial_fit(self, X, y=None):
        if self.n_samples_seen_ == 0:  # first fit
            return self.fit(X)
        lifted_a = self.n_samples_seen_, self.feature_names
        lifted_b = self._lift(X, self._hyperparams)
        self._set_fit_attributes(self._combine(lifted_a, lifted_b))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer(X)
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        self.n_samples_seen_, self.feature_names = lifted
        self._transformer = None

    def _build_transformer(self, X):
        cols = self._hyperparams["cols"]
        hash_method = self._hyperparams["hash_method"]
        N = self._hyperparams["n_components"]
        columns_cat = {
            f"col_{i}": reduce(
                Expr.__add__,  # type: ignore
                [
                    ite(hash(hash_method, it[col_name]) % N == i, 1, 0)  # type: ignore
                    for col_name in cols
                ],
            )
            for i in range(N)
        }
        columns_num = {col: it[col] for col in get_columns(X) if col not in cols}
        result = Map(columns={**columns_cat, **columns_num})
        return result

    @staticmethod
    def _lift(X, hyperparams):
        if hyperparams["cols"] is None:
            hyperparams["cols"] = get_obj_cols(X)
        cols = hyperparams["cols"]
        N = hyperparams["n_components"]
        feature_names_cat = [f"col_{i}" for i in range(N)]
        feature_names_num = [col for col in get_columns(X) if col not in cols]
        feature_names = feature_names_cat + feature_names_num
        n_samples_seen_ = count(X)
        return n_samples_seen_, feature_names

    @staticmethod
    def _combine(lifted_a, lifted_b):
        n_samples_seen_a, feature_names_a = lifted_a
        n_samples_seen_b, feature_names_b = lifted_b
        assert list(feature_names_a) == list(feature_names_b)
        n_samples_seen_ = n_samples_seen_a + n_samples_seen_b
        return n_samples_seen_, feature_names_a


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
