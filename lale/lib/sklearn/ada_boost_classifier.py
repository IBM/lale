# Copyright 2019 IBM Corporation
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

from sklearn.ensemble import AdaBoostClassifier as SKLModel

import lale.docstrings
import lale.operators


class _AdaBoostClassifierImpl:
    def __init__(
        self,
        base_estimator=None,
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=None,
    ):
        estimator_impl = base_estimator
        if isinstance(estimator_impl, lale.operators.Operator):
            if isinstance(estimator_impl, lale.operators.IndividualOp):
                estimator_impl = estimator_impl._impl_instance()
                wrapped_model = getattr(estimator_impl, "_wrapped_model", None)
                if wrapped_model is not None:
                    estimator_impl = wrapped_model
            else:
                raise ValueError(
                    "If base_estimator is a Lale operator, it needs to be an individual operator. "
                )
        self._hyperparams = {
            "base_estimator": estimator_impl,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "algorithm": algorithm,
            "random_state": random_state,
        }
        self._wrapped_model = SKLModel(**self._hyperparams)
        self._hyperparams["base_estimator"] = base_estimator

    def get_params(self, deep=True):
        out = self._wrapped_model.get_params(deep=deep)
        # we want to return the lale operator, not the underlying impl
        out["base_estimator"] = self._hyperparams["base_estimator"]
        return out

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)

    def score(self, X, y, sample_weight=None):
        return self._wrapped_model.score(X, y, sample_weight)


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
                "n_estimators",
                "learning_rate",
                "algorithm",
                "random_state",
            ],
            "relevantToOptimizer": ["n_estimators", "learning_rate", "algorithm"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                    "description": "The base estimator from which the boosted ensemble is built.",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 50,
                    "maximumForOptimizer": 500,
                    "distribution": "uniform",
                    "default": 50,
                    "description": "The maximum number of estimators at which boosting is terminated.",
                },
                "learning_rate": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1.0,
                    "description": "Learning rate shrinks the contribution of each classifier by",
                },
                "algorithm": {
                    "enum": ["SAMME", "SAMME.R"],
                    "default": "SAMME.R",
                    "description": "If 'SAMME.R' then use the SAMME.R real boosting algorithm.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator;",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Build a boosted classifier from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "The target values (class labels).",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Sample weights. If None, the sample weights are initialized to",
        },
    },
}
_input_predict_schema = {
    "description": "Predict classes for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
    },
}
_output_predict_schema = {
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Predict class probabilities for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
    },
}
_output_predict_proba_schema = {
    "description": "The class probabilities of the input samples. The order of",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}

_input_decision_function_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_decision_function_schema = {
    "description": "Confidence scores for samples for each class in the model.",
    "anyOf": [
        {
            "description": "In the multi-way case, score per (sample, class) combination.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        {
            "description": "In the binary case, score for `self._classes[1]`.",
            "type": "array",
            "items": {"type": "number"},
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`AdaBoost classifier`_ from scikit-learn for boosting ensemble.

.. _`AdaBoost classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ada_boost_classifier.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

AdaBoostClassifier = lale.operators.make_operator(
    _AdaBoostClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(AdaBoostClassifier)
