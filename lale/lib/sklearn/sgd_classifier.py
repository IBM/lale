# Copyright 2019-2022 IBM Corporation
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

import sklearn
from sklearn.linear_model import SGDClassifier as SKLModel

import lale.docstrings
import lale.operators

from ._common_schemas import (
    schema_1D_cats,
    schema_2D_numbers,
    schema_sample_weight,
    schema_X_numbers,
)

_hyperparams_schema = {
    "description": "inherited docstring for SGDClassifier Linear classifiers (SVM, logistic regression, a.o.) with SGD training.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "loss",
                "penalty",
                "alpha",
                "l1_ratio",
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "verbose",
                "epsilon",
                "n_jobs",
                "random_state",
                "learning_rate",
                "eta0",
                "power_t",
                "early_stopping",
                "validation_fraction",
                "n_iter_no_change",
                "class_weight",
                "warm_start",
                "average",
            ],
            "relevantToOptimizer": [
                "loss",
                "penalty",
                "alpha",
                "l1_ratio",
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "epsilon",
                "learning_rate",
                "eta0",
                "power_t",
            ],
            "additionalProperties": False,
            "properties": {
                "loss": {
                    "enum": [
                        "hinge",
                        "log",
                        "modified_huber",
                        "squared_hinge",
                        "perceptron",
                        "squared_loss",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                    "default": "hinge",
                    "description": "The loss function to be used. Defaults to 'hinge', which gives a linear SVM.",
                },
                "penalty": {
                    "description": "The penalty (aka regularization term) to be used. Defaults to 'l2'",
                    "enum": ["elasticnet", "l1", "l2"],
                    "default": "l2",
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Constant that multiplies the regularization term. Defaults to 0.0001",
                },
                "l1_ratio": {
                    "type": "number",
                    "minimumForOptimizer": 1e-9,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.15,
                    "description": "The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the intercept should be estimated or not. If False, the",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
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
                    "default": 0.001,
                    "description": "The stopping criterion.",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not the training data should be shuffled after each epoch.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "The verbosity level",
                },
                "epsilon": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 0.1,
                    "description": "Epsilon in the epsilon-insensitive loss functions; only if `loss` is",
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
                "learning_rate": {
                    "enum": ["optimal", "constant", "invscaling", "adaptive"],
                    "default": "optimal",
                    "description": "The learning rate schedule:",
                },
                "eta0": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0,
                    "description": "The initial learning rate for the 'constant', 'invscaling' or",
                },
                "power_t": {
                    "type": "number",
                    "minimumForOptimizer": 0.00001,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 0.5,
                    "description": "The exponent for inverse scaling learning rate [default 0.5].",
                },
                "early_stopping": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use early stopping to terminate training when validation",
                },
                "validation_fraction": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
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
                "class_weight": {
                    "anyOf": [{"type": "object"}, {"enum": ["balanced", None]}],
                    "default": None,
                    "description": "Preset for the class_weight fit parameter.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit as",
                },
                "average": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer", "forOptimizer": False},
                    ],
                    "default": False,
                    "description": "When set to True, computes the averaged SGD weights and stores the result in the ``coef_`` attribute.",
                },
            },
        },
        {
            "description": "eta0 must be greater than 0 if the learning_rate is not ‘optimal’.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"enum": ["optimal"]},
                    },
                },
                {
                    "type": "object",
                    "properties": {
                        "eta0": {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                        },
                    },
                },
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Fit linear model with Stochastic Gradient Descent.",
    "required": ["X", "y"],
    "type": "object",
    "properties": {
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "coef_init": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The initial coefficients to warm-start the optimization.",
        },
        "intercept_init": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The initial intercept to warm-start the optimization.",
        },
        "sample_weight": schema_sample_weight,
    },
}

_input_partial_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "classes": schema_1D_cats,
        "sample_weight": schema_sample_weight,
    },
}

_output_predict_proba_schema = {
    "description": "Returns the probability of the sample for each class in the model,",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
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
    "description": """`SGD classifier`_ from scikit-learn uses linear classifiers (SVM, logistic regression, a.o.) with stochastic gradient descent training.

.. _`SGD classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.sgd_classifier.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_partial_fit": _input_partial_fit_schema,
        "input_predict": schema_X_numbers,
        "output_predict": schema_1D_cats,
        "input_predict_proba": schema_X_numbers,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": schema_X_numbers,
        "output_decision_function": _output_decision_function_schema,
    },
}


SGDClassifier = lale.operators.make_operator(SKLModel, _combined_schemas)

if sklearn.__version__ >= "1.0":
    # old: https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.SGDClassifer.html
    # new: https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.SGDClassifier.html
    SGDClassifier = SGDClassifier.customize_schema(
        loss={
            "description": """The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.
The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’,
or a regression loss: ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
The ‘log’ loss gives logistic regression, a probabilistic classifier.
‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates.
‘squared_hinge’ is like hinge but is quadratically penalized.
‘perceptron’ is the linear loss used by the perceptron algorithm.
The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
More details about the losses formulas can be found in the scikit-learn User Guide.""",
            "anyOf": [
                {
                    "enum": [
                        "hinge",
                        "log",
                        "modified_huber",
                        "squared_hinge",
                        "perceptron",
                        "squared_error",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                },
                {"enum": ["squared_loss"], "forOptimizer": False},
            ],
            "default": "hinge",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(SGDClassifier)
