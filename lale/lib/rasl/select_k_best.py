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
from scipy import special

import lale.docstrings
import lale.operators
from lale.expressions import count as agg_count
from lale.expressions import it
from lale.expressions import sum as agg_sum
from lale.helpers import _ensure_pandas
from lale.lib.dataframe import count, get_columns
from lale.lib.lale.concat_features import ConcatFeatures
from lale.lib.rasl import Aggregate, GroupBy, Map
from lale.lib.sklearn import select_k_best

from ._monoid import Monoid
from .scores import FClassif


class _SelectKBestMonoid(Monoid):
    def __init__(self, *, TODO):
        pass


class _SelectKBestImpl:
    def __init__(
        self,
        monoidable_score_func=FClassif,
        score_func=None,
        *,
        k=10
    ):
        self._hyperparams = {
            "score_func": monoidable_score_func(),
            "k": k,
        }
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, y, self._hyperparams))
        return self

    def partial_fit(self, X, y=None):
        if self.n_samples_seen_ == 0:  # first fit
            return self.fit(X, y)
        lifted_a = (self.n_samples_seen_, self.feature_names_in_, self.lifted_score_)
        lifted_b = self._lift(X, y, self._hyperparams)
        self._set_fit_attributes(self._combine(lifted_a, lifted_b, self._hyperparams))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        self.n_samples_seen_, self.feature_names_in_, self.lifted_score_ = lifted
        score_func = self._hyperparams["score_func"]
        self.scores_, self.pvalues_ = score_func._from_monoid(self.lifted_score_)
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
        n_samples_seen = count(X)
        feature_names_in = get_columns(X)
        lifted_score = score_func._to_monoid((X, y))
        return n_samples_seen, feature_names_in, lifted_score

    @staticmethod
    def _combine(lifted_a, lifted_b, hyperparams):
        (n_samples_seen_a, feature_names_in_a, lifted_score_a) = lifted_a
        (n_samples_seen_b, feature_names_in_b, lifted_score_b) = lifted_b
        n_samples_seen = n_samples_seen_a + n_samples_seen_b
        assert list(feature_names_in_a) == list(feature_names_in_b)
        feature_names_in = feature_names_in_a
        lifted_score = lifted_score_a.combine(lifted_score_b)
        return n_samples_seen, feature_names_in, lifted_score


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
