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

from sklearn.ensemble import BaggingClassifier as SKLModel

import lale.docstrings
import lale.operators
from lale.sklearn_compat import make_sklearn_compat


class BaggingClassifierImpl:
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self._hyperparams = {
            "base_estimator": make_sklearn_compat(base_estimator),
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "bootstrap_features": bootstrap_features,
            "oob_score": oob_score,
            "warm_start": warm_start,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
        }
        self._wrapped_model = SKLModel(**self._hyperparams)

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


_hyperparams_schema = {
    "description": "A Bagging classifier.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
                "n_estimators",
                "max_samples",
                "max_features",
                "bootstrap",
                "bootstrap_features",
                "oob_score",
                "warm_start",
                "n_jobs",
                "random_state",
                "verbose",
            ],
            "relevantToOptimizer": ["n_estimators", "bootstrap"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                    "description": "The base estimator to fit on random subsets of the dataset.",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 10,
                    "description": "The number of base estimators in the ensemble.",
                },
                "max_samples": {
                    "description": "The number of samples to draw from X to train each base estimator.",
                    "anyOf": [
                        {
                            "description": "Draw max_samples samples.",
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/maxItems",  # number of rows
                        },
                        {
                            "description": "Draw max_samples * X.shape[0] samples.",
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                        },
                    ],
                    "default": 1.0,
                },
                "max_features": {
                    "description": "The number of features to draw from X to train each base estimator.",
                    "anyOf": [
                        {
                            "description": "Draw max_features features.",
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                        },
                        {
                            "description": "Draw max_samples * X.shape[1] features.",
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                        },
                    ],
                    "default": 1.0,
                },
                "bootstrap": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether samples are drawn with replacement. If False, sampling",
                },
                "bootstrap_features": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether features are drawn with replacement.",
                },
                "oob_score": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use out-of-bag samples to estimate",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit",
                },
                "n_jobs": {
                    "description": "The number of jobs to run in parallel for both `fit` and `predict`.",
                    "anyOf": [
                        {
                            "description": "1 unless in joblib.parallel_backend context.",
                            "enum": [None],
                        },
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of CPU cores.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": None,
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
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Controls the verbosity when fitting and predicting.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Build a Bagging ensemble of estimators from the training",
    "type": "object",
    "required": ["y", "X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "The training input samples. Sparse matrices are accepted only if",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The target values (class labels in classification, real numbers in",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"},},
                {"enum": [None]},
            ],
            "description": "Sample weights. If None, then samples are equally weighted.",
        },
    },
}
_input_predict_schema = {
    "description": "Predict class for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "The training input samples. Sparse matrices are accepted only if",
        },
    },
}
_output_predict_schema = {
    "description": "The predicted classes.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "description": "Predict class probabilities for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "The training input samples. Sparse matrices are accepted only if",
        },
    },
}
_output_predict_proba_schema = {
    "description": "The class probabilities of the input samples. The order of the",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"},},
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
    "description": """`Bagging classifier`_ from scikit-learn for bagging ensemble.

.. _`Bagging classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.bagging_classifier.html",
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

lale.docstrings.set_docstrings(BaggingClassifierImpl, _combined_schemas)

BaggingClassifier = lale.operators.make_operator(
    BaggingClassifierImpl, _combined_schemas
)
