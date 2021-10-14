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
    it,
    max,
    min,
)
from lale.lib.lale import Aggregate
class _MinMaxScalerImpl:
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
      self.feature_range = feature_range
      self.copy = copy
      self.clip = clip

    def fit(self, X, y=None):
      # import pdb; pdb.set_trace()
      agg = { f'{c}_min': min(it[c]) for c in X.columns }
      agg.update({ f'{c}_max': max(it[c]) for c in X.columns })
      aggregate = Aggregate(columns=agg)
      min_max = aggregate.transform(X)
      print("XXXXXXXXX", min_max)
      pass

    def transform(self, X):
      pass


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
