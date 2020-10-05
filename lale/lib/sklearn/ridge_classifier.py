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

import sklearn.linear_model

import lale.docstrings
import lale.operators


class RidgeClassifierImpl:
    def __init__(
        self,
        alpha=None,
        fit_intercept=None,
        normalize=False,
        copy_X=True,
        max_iter=None,
        tol=0.001,
        solver=None,
        class_weight=None,
        random_state=None,
    ):
        self._hyperparams = {
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "normalize": normalize,
            "copy_X": copy_X,
            "max_iter": max_iter,
            "tol": tol,
            "solver": solver,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        self._wrapped_model = sklearn.linear_model.RidgeClassifier(**self._hyperparams)

    def fit(self, X, y, **fit_params):
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)


_hyperparams_schema = {
    "description": "Classifier using Ridge regression.",
    "allOf": [
        {
            "type": "object",
            "required": ["alpha", "fit_intercept", "solver"],
            "relevantToOptimizer": [
                "alpha",
                "fit_intercept",
                "normalize",
                "copy_X",
                "max_iter",
                "tol",
                "solver",
            ],
            "additionalProperties": False,
            "properties": {
                "alpha": {
                    "description": "Regularization strength; larger values specify stronger regularization.",
                    "anyOf": [
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "minimumForOptimizer": 1e-05,
                            "maximumForOptimizer": 10.0,
                            "distribution": "loguniform",
                        },
                        {
                            "type": "array",
                            "description": "Penalties specific to the targets.",
                            "items": {
                                "type": "number",
                                "minimum": 0.0,
                                "exclusiveMinimum": True,
                            },
                            "forOptimizer": False,
                        },
                    ],
                    "default": 1.0,
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model.",
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False.",
                },
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, X will be copied; else, it may be overwritten.",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 10,
                            "maximumForOptimizer": 1000,
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Maximum number of iterations for conjugate gradient solver.",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.001,
                    "description": "Precision of the solution.",
                },
                "solver": {
                    "enum": [
                        "auto",
                        "svd",
                        "cholesky",
                        "lsqr",
                        "sparse_cg",
                        "sag",
                        "saga",
                    ],
                    "default": "auto",
                    "description": "Solver to use in the computational routines.",
                },
                "class_weight": {
                    "anyOf": [
                        {"type": "object"},  # dict, list of dicts,
                        {"enum": ["balanced", None]},
                    ],
                    "description": "Weights associated with classes in the form ``{class_label: weight}``.",
                    "default": None,
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator to use when shuffling",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit Ridge regression model",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "Training data",
        },
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"},},
                },
                {"type": "array", "items": {"type": "number"},},
                {"type": "array", "items": {"type": "string"},},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "Target values",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "number"},
                {"type": "array", "items": {"type": "number"},},
                {"enum": [None]},
            ],
            "description": "Individual weights for each sample",
        },
    },
}
_input_predict_schema = {
    "description": "Predict class labels for samples in X.",
    "type": "object",
    "properties": {
        "X": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"},},
                },
            ],
            "description": "Samples.",
        },
    },
}
_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
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
    "description": """`Ridge classifier`_ from scikit-learn.

.. _`Ridge classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ridge_classifier.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

lale.docstrings.set_docstrings(RidgeClassifierImpl, _combined_schemas)

RidgeClassifier = lale.operators.make_operator(RidgeClassifierImpl, _combined_schemas)
