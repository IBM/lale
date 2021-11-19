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

import lale.docstrings
import lale.operators
from lale.lib.sklearn import min_max_scaler
from lale.expressions import (
    Expr,
    it,
    max,
    min,
    subtract,
    ratio,
)
from lale.lib.lale import (
  Aggregate,
  Map,
)

import ast
from lale.helpers import (
    _is_pandas_df,
    _is_spark_df,
)

class _MinMaxScalerImpl:
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
      self.feature_range = feature_range
      self.copy = copy
      self.clip = clip
      self.data_min_ = None
      self.data_max_ = None

    def fit(self, X, y=None):
      agg = { f'{c}_min': min(it[c]) for c in X.columns }
      agg.update({ f'{c}_max': max(it[c]) for c in X.columns })
      aggregate = Aggregate(columns=agg)
      data_min_max = aggregate.transform(X)
      self.data_min_max_ = data_min_max
      if _is_pandas_df(X):
        self.data_min_ = data_min_max.loc['min'].values # how to make that attribute of the outer object?
        self.data_max_ = data_min_max.loc['max'].values
      elif _is_spark_df(X):
        # TODO
        pass
      else:
        raise ValueError(
          "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
        )
      self.data_range_ = self.data_max_ - self.data_min_
      self.n_features_in_ = len(X.columns)
      self.feature_names_in_ = X.columns
      return self # should it be a copy?

    def transform(self, X):
      ops = {}
      for c in X.columns:
        min, max = self.feature_range
        if _is_pandas_df(X):
          X_min = self.data_min_max_[c]['min']
          X_max = self.data_min_max_[c]['max']
        elif _is_spark_df(X):
          # TODO
          pass
        else:
          raise ValueError(
            "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
          )
        op = (it[c] - Expr(ast.Num(X_min))) / Expr(ast.Num(X_max - X_min)) # TODO: the expression language is not expressive enough to handle this case
        ops.update({ f'{c}_scaled': op  })
      transformer = Map(columns=ops).fit(X)
      X_transformed = transformer.transform(X)
      return X_transformed

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra implementation of MinMaxScaler.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.min_max_scaler.html",
    "type": "object",
        "tags": {
        "pre": ["~categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": min_max_scaler._hyperparams_schema,
        "input_fit": min_max_scaler._input_schema_fit,
        "input_transform": min_max_scaler._input_transform_schema,
        "output_transform": min_max_scaler._output_transform_schema,
    },
}

MinMaxScaler = lale.operators.make_operator(_MinMaxScalerImpl, _combined_schemas)

lale.docstrings.set_docstrings(MinMaxScaler)
