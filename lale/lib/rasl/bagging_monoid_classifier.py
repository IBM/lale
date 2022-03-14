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

import typing

import numpy as np

import lale.docstrings
import lale.operators
from lale.datasets.data_schemas import forward_metadata
from lale.expressions import it
from lale.expressions import max as agg_max
from lale.expressions import min as agg_min
from lale.helpers import _is_spark_df
from lale.lib.dataframe import count, get_columns
from lale.lib.rasl import Aggregate, Map
from lale.lib.sklearn import bagging_classifier
from lale.schemas import Enum
from lale.lib.sklearn import DecisionTreeClassifier
from .monoid import Monoid, MonoidableOperator


class _BaggingClassifierMonoid(Monoid):
    def __init__(
        self,
        classifiers
    ):
        self.classifiers = classifiers

    def combine(self, other):
        return _BaggingClassifierMonoid(
            classifiers=self.classifiers.extend(other.classifiers)
        )

class _BaggingMonoidClassifierImpl(MonoidableOperator[_BaggingClassifierMonoid]):
    def __init__(self, base_estimator=None):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        self._hyperparams = {"base_estimator": base_estimator}

    def predict(self, X):
        if len(self.classifiers_list)==1:
            return self.classifiers_list[0].predict(X)
        else:
            #TODO: Take a voting of the classifiers
            pass

    def from_monoid(self, v: _BaggingClassifierMonoid):
        self.classifiers_list = v.classifiers

    def to_monoid(self, v) -> _BaggingClassifierMonoid:
        X, y = v
        trainable = self._hyperparams["base_estimator"]
        trained_classifier = trainable.fit(X, y)
        return _BaggingClassifierMonoid(
            [trained_classifier]
        )


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
        "hyperparams": bagging_classifier._hyperparams_schema,
        "input_fit": bagging_classifier._input_fit_schema,
        "input_transform": bagging_classifier.schema_X_numbers,
        "output_transform": bagging_classifier.schema_1D_cats,
    },
}

BaggingMonoidClassifier = lale.operators.make_operator(_BaggingMonoidClassifierImpl, _combined_schemas)
lale.docstrings.set_docstrings(BaggingMonoidClassifier)
