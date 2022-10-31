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

import sklearn.linear_model

import lale.docstrings
import lale.operators

from ._common_schemas import (
    schema_1D_cats,
    schema_2D_numbers,
    schema_sample_weight,
    schema_X_numbers,
)

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "penalty",
                "alpha",
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "verbose",
                "eta0",
                "n_jobs",
                "random_state",
                "early_stopping",
                "validation_fraction",
                "n_iter_no_change",
                "class_weight",
                "warm_start",
            ],
            "relevantToOptimizer": [
                "alpha",
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "eta0",
            ],
            "additionalProperties": False,
            "properties": {
                "penalty": {
                    "enum": ["l2", "l1", "elasticnet", None],
                    "description": "The penalty (aka regularization term) to be used.",
                    "default": None,
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Constant that multiplies the regularization term if regularization is used.",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 10000,
                    "distribution": "loguniform",
                    "default": 1000,
                    "description": "The maximum number of passes over the training data (aka epochs).",
                },
                "tol": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 1e-08,
                            "maximumForOptimizer": 0.01,
                            "description": "If not None, the iterations will stop when (loss > previous_loss - tol).",
                        },
                        {"enum": [None]},
                    ],
                    "default": 1e-3,
                    "description": "The stopping criterion",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not the training data should be shuffled after each epoch.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "The verbosity level.",
                },
                "eta0": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1.0,
                    "description": "Constant by which the updates are multiplied.",
                },
                "n_jobs": {
                    "description": "The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation.",
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
                "early_stopping": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use early stopping to terminate training when validation score is not improving.",
                },
                "validation_fraction": {
                    "type": "number",
                    "default": 0.1,
                    "minimum": 0,
                    "maximum": 1,
                    "description": "The proportion of training data to set aside as validation set for early stopping.",
                },
                "n_iter_no_change": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of iterations with no improvement to wait before early stopping.",
                },
                "class_weight": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": {"type": "number"}},
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                        },
                        {"enum": ["balanced", None]},
                    ],
                    "description": "Weights associated with classes in the form ``{class_label: weight}``.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.",
                },
            },
        },
    ],
}

_input_fit_schema = {
    "description": "Fit linear model with Stochastic Gradient Descent.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "coef_init": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {"enum": [None]},
            ],
            "description": "The initial coefficients to warm-start the optimization.",
        },
        "intercept_init": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None]},
            ],
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

_output_decision_function_schema = {
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
    "description": """`Perceptron`_ classifier from scikit-learn.

.. _`Perceptron`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.perceptron.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_partial_fit": _input_partial_fit_schema,
        "input_predict": schema_X_numbers,
        "output_predict": schema_1D_cats,
        "input_decision_function": schema_X_numbers,
        "output_decision_function": _output_decision_function_schema,
    },
}

Perceptron: lale.operators.PlannedIndividualOp
Perceptron = lale.operators.make_operator(
    sklearn.linear_model.Perceptron, _combined_schemas
)

lale.docstrings.set_docstrings(Perceptron)
