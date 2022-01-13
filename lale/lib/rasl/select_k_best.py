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

import numpy as np
from sklearn.feature_selection import f_classif

import lale.docstrings
import lale.operators
from lale.expressions import it
from lale.lib.rasl import Map
from lale.lib.sklearn import select_k_best


class _SelectKBestImpl:
    def __init__(self, score_func=f_classif, *, k=10):
        self._hyperparams = {"score_func": score_func, "k": k}

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, y, self._hyperparams))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        self.feature_names_in_, self.scores_, self.pvalues_ = lifted
        self.n_features_in_ = len(self.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        k = self._hyperparams["k"]
        scores = self.scores_.copy()
        scores[np.isnan(scores)] = np.finfo(scores.dtype).min
        ind = np.sort(np.argpartition(scores, -k)[-k:])
        kbest = self.feature_names_in_[ind]
        result = Map(columns={col: it[col] for col in kbest})
        return result

    @staticmethod
    def _lift(X, y, hyperparams):
        score_func = hyperparams["score_func"]
        feature_names_in = X.columns
        scores, pvalues = score_func(X, y)
        return feature_names_in, scores, pvalues


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra implementation of SelectKBest.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.select_k_best.html",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": select_k_best._hyperparams_schema,
        "input_fit": select_k_best._input_fit_schema,
        "input_transform": select_k_best._input_transform_schema,
        "output_transform": select_k_best._output_transform_schema,
    },
}

SelectKBest: lale.operators.PlannedIndividualOp
SelectKBest = lale.operators.make_operator(_SelectKBestImpl, _combined_schemas)

lale.docstrings.set_docstrings(SelectKBest)
