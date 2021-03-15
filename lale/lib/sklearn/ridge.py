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

_hyperparams_schema = {
    "description": "Linear least squares with l2 regularization.",
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
                            "minimumForOptimizer": 1e-10,
                            "maximumForOptimizer": 1.0,
                            "default": 1.0,
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
                            "minimum": 1,
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
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Training data",
        },
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
            ],
            "description": "Target values",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "number"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "description": "Individual weights for each sample",
        },
    },
}
_input_predict_schema = {
    "description": "Predict using the linear model",
    "type": "object",
    "properties": {
        "X": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
            ],
            "description": "Samples.",
        },
    },
}
_output_predict_schema = {
    "description": "Returns predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {  # There was a case where Ridge returned 2-d predictions for a single target.
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Ridge`_ regression estimator from scikit-learn.

.. _`Ridge`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ridge.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


Ridge = lale.operators.make_operator(sklearn.linear_model.Ridge, _combined_schemas)

lale.docstrings.set_docstrings(Ridge)
