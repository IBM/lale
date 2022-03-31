# Copyright 2022 IBM Corporation
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

import copy
import inspect
from typing import Any, Tuple

import pandas as pd
import sklearn.base

import lale.docstrings
import lale.operators
from lale.lib._common_schemas import schema_estimator
from lale.lib.sklearn import DecisionTreeClassifier, bagging_classifier

from .monoid import Monoid, MonoidableOperator


class _BatchedBaggingClassifierMonoid(Monoid):
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def combine(self, other: "_BatchedBaggingClassifierMonoid"):
        orig_classifiers = copy.copy(self.classifiers)
        orig_classifiers.extend(other.classifiers)
        return _BatchedBaggingClassifierMonoid(classifiers=orig_classifiers)


class _BatchedBaggingClassifierImpl(
    MonoidableOperator[_BatchedBaggingClassifierMonoid]
):
    def __init__(self, base_estimator=None):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        self._hyperparams = {"base_estimator": base_estimator}

    def predict(self, X):
        if len(self.classifiers_list) == 1:
            return self.classifiers_list[0].predict(X)
        else:
            # Take a voting of the classifiers
            predictions_list = [
                classifier.predict(X) for classifier in self.classifiers_list
            ]
            df = pd.DataFrame(predictions_list).transpose()
            predictions = df.mode(axis=1)
            if (
                predictions.shape[1] > 1
            ):  # When there are multiple modes, pick the first one
                predictions = predictions.iloc[:, 0]
            predictions = predictions.squeeze()  # converts a dataframe to series.
            return predictions

    def from_monoid(self, v: _BatchedBaggingClassifierMonoid):
        self._monoid = v
        self.classifiers_list = v.classifiers

    def to_monoid(self, v: Tuple[Any, Any]) -> _BatchedBaggingClassifierMonoid:
        X, y = v
        trainable = self._hyperparams["base_estimator"]
        if isinstance(trainable, sklearn.base.BaseEstimator):
            trainable = sklearn.base.clone(trainable)
        if inspect.isclass(trainable):
            trainable = trainable()
        trained_classifier = trainable.fit(X, y)
        return _BatchedBaggingClassifierMonoid([trained_classifier])


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
            ],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "base_estimator": schema_estimator,
            },
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Implementation of a homomorphic bagging classifier.
    As proposed in https://izbicki.me/public/papers/icml2013-algebraic-classifiers.pdf""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.bagging_monoid_classifier.html",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["estimator"],
        "post": [],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": bagging_classifier._input_fit_schema,
        "input_predict": bagging_classifier.schema_X_numbers,
        "output_predict": bagging_classifier.schema_1D_cats,
    },
}

BatchedBaggingClassifier = lale.operators.make_operator(
    _BatchedBaggingClassifierImpl, _combined_schemas
)
lale.docstrings.set_docstrings(BatchedBaggingClassifier)
