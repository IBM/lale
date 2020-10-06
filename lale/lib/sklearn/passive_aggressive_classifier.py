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

# old: https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
# new: https://scikit-learn.org/0.23/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
from lale.schemas import Int


class PassiveAggressiveClassifierImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.linear_model.PassiveAggressiveClassifier(
            **self._hyperparams
        )

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)


_hyperparams_schema = {
    "description": "Passive Aggressive Classifier",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "C",
                "fit_intercept",
                "max_iter",
                "tol",
                "early_stopping",
                "shuffle",
                "loss",
                "average",
            ],
            "relevantToOptimizer": [
                "C",
                "fit_intercept",
                "max_iter",
                "tol",
                "early_stopping",
                "shuffle",
                "loss",
                "average",
            ],
            "properties": {
                "C": {
                    "type": "number",
                    "description": "Maximum step size (regularization). Defaults to 1.0.",
                    "default": 1.0,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-5,
                    "maximumForOptimizer": 10,
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the intercept should be estimated or not. If False, the"
                    "the data is assumed to be already centered.",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 5,
                            "maximumForOptimizer": 1000,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": 5,
                    "description": "The maximum number of passes over the training data (aka epochs).",
                },
                "tol": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 1e-08,
                            "maximumForOptimizer": 0.01,
                            "distribution": "loguniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,  # default value is 1e-3 from sklearn 0.21.
                    "description": "The stopping criterion. If it is not None, the iterations will stop",
                },
                "early_stopping": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use early stopping to terminate training when validation.",
                },
                "validation_fraction": {
                    "type": "number",
                    "default": 0.1,
                    "description": "The proportion of training data to set aside as validation set for",
                },
                "n_iter_no_change": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 10,
                    "default": 5,
                    "description": "Number of iterations with no improvement to wait before early stopping.",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not the training data should be shuffled after each epoch.",
                },
                "verbose": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 0,
                    "description": "The verbosity level",
                },
                "loss": {
                    "enum": ["hinge", "squared_hinge"],
                    "default": "hinge",
                    "description": "The loss function to be used:",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of CPUs to use to do the OVA (One Versus All, for",
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
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to"
                    " fit as initialization, otherwise, just erase the previous solution.",
                },
                "class_weight": {
                    "anyOf": [{"type": "object"}, {"enum": ["balanced", None]}],
                    "default": None,
                    "description": "Preset for the class_weight fit parameter.",
                },
                "average": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer", "forOptimizer": False},
                    ],
                    "default": False,
                    "description": "When set to True, computes the averaged SGD weights and stores the",
                },
                "n_iter": {
                    "anyOf": [{"type": "integer", "minimum": 1}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of passes over the training data (aka epochs).",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit linear model with Passive Aggressive algorithm.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "description": "Training data",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target values",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
        "coef_init": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The initial coefficients to warm-start the optimization.",
        },
        "intercept_init": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The initial intercept to warm-start the optimization.",
        },
    },
}
_input_predict_schema = {
    "description": "Predict class labels for samples in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "description": "Test data",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
}
_output_predict_schema = {
    "description": "Predict class labels for samples in X.",
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
    "description": """`Passive aggressive`_ classifier from scikit-learn.

.. _`Passive aggressive`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.passive_aggressive_classifier.html",
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

PassiveAggressiveClassifier: lale.operators.IndividualOp
PassiveAggressiveClassifier = lale.operators.make_operator(
    PassiveAggressiveClassifierImpl, _combined_schemas
)


if sklearn.__version__ >= "0.21":
    PassiveAggressiveClassifier = PassiveAggressiveClassifier.customize_schema(
        max_iter=Int(
            minForOptimizer=5,
            maxForOptimizer=1000,
            distribution="uniform",
            desc="The maximum number of passes over the training data (aka epochs).",
            default=1000,
        )
    )

if sklearn.__version__ >= "0.22":
    PassiveAggressiveClassifier = PassiveAggressiveClassifier.customize_schema(
        n_iter=None
    )

lale.docstrings.set_docstrings(
    PassiveAggressiveClassifierImpl, PassiveAggressiveClassifier._schemas
)
