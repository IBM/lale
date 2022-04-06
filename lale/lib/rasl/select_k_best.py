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
from typing import Any, Tuple

import numpy as np

import lale.docstrings
import lale.operators
from lale.expressions import it
from lale.lib.dataframe import count, get_columns
from lale.lib.rasl import Map
from lale.lib.sklearn import select_k_best

from .monoid import Monoid, MonoidableOperator
from .scores import FClassif


class _SelectKBestMonoid(Monoid):
    def __init__(self, *, n_samples_seen_, feature_names_in_, lifted_score_):
        self.n_samples_seen_ = n_samples_seen_
        self.feature_names_in_ = feature_names_in_
        self.lifted_score_ = lifted_score_

    def combine(self, other: "_SelectKBestMonoid"):
        n_samples_seen_ = self.n_samples_seen_ + other.n_samples_seen_
        assert list(self.feature_names_in_) == list(other.feature_names_in_)
        feature_names_in_ = self.feature_names_in_
        lifted_score_ = self.lifted_score_.combine(other.lifted_score_)
        return _SelectKBestMonoid(
            n_samples_seen_=n_samples_seen_,
            feature_names_in_=feature_names_in_,
            lifted_score_=lifted_score_,
        )


class _SelectKBestImpl(MonoidableOperator[_SelectKBestMonoid]):
    def __init__(self, monoidable_score_func=FClassif, score_func=None, *, k=10):
        self._hyperparams = {
            "score_func": monoidable_score_func(),
            "k": k,
        }

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    @property
    def n_samples_seen_(self):
        return getattr(self._monoid, "n_samples_seen_", 0)

    @property
    def feature_names_in_(self):
        return getattr(self._monoid, "feature_names_in_", None)

    def from_monoid(self, lifted):
        self._monoid = lifted
        score_func = self._hyperparams["score_func"]
        lifted_score_ = self._monoid.lifted_score_
        self.scores_, self.pvalues_ = score_func.from_monoid(lifted_score_)
        self.n_features_in_ = len(self._monoid.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        assert self._monoid is not None
        k = self._hyperparams["k"]
        scores = self.scores_.copy()
        scores[np.isnan(scores)] = np.finfo(scores.dtype).min
        ind = np.sort(np.argpartition(scores, -min(k, len(scores)))[-k:])
        kbest = self._monoid.feature_names_in_[ind]
        result = Map(columns={col: it[col] for col in kbest})
        return result

    def to_monoid(self, v: Tuple[Any, Any]):
        X, y = v
        score_func = self._hyperparams["score_func"]
        n_samples_seen_ = count(X)
        feature_names_in_ = get_columns(X)
        lifted_score_ = score_func.to_monoid((X, y))
        return _SelectKBestMonoid(
            n_samples_seen_=n_samples_seen_,
            feature_names_in_=feature_names_in_,
            lifted_score_=lifted_score_,
        )


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
