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

import sklearn
import sklearn.linear_model

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": ["fit_intercept", "normalize", "copy_X"],
            "relevantToOptimizer": ["fit_intercept", "normalize", "copy_X"],
            "additionalProperties": False,
            "properties": {
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model.",
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.",
                },
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, X will be copied; else, it may be overwritten.",
                },
                "n_jobs": {
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
                    "description": "The number of jobs to run in parallel.",
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {"type": "array", "items": {"type": "number"}},
            ],
            "description": "Target values. Will be cast to X's dtype if necessary",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "Samples are equally weighted."},
            ],
            "description": "Sample weights.",
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Samples.",
        }
    },
}

_output_predict_schema = {
    "description": "Returns predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        {"type": "array", "items": {"type": "number"}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Linear regression`_ linear model from scikit-learn for classification.

.. _`Linear regression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.linear_regression.html",
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

LinearRegression: lale.operators.PlannedIndividualOp
LinearRegression = lale.operators.make_operator(
    sklearn.linear_model.LinearRegression, _combined_schemas
)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.LinearRegression.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.LinearRegression.html
    LinearRegression = LinearRegression.customize_schema(
        positive={
            "type": "boolean",
            "description": "When set to True, forces the coefficients to be positive.",
            "default": False,
        }
    )
    LinearRegression = LinearRegression.customize_schema(
        constraint={
            "description": "Setting positive=True is only supported for dense arrays.",
            "anyOf": [
                {"type": "object", "properties": {"positive": {"enum": [False]}}},
                {"type": "object", "laleNot": "X/isSparse"},
            ],
        }
    )


lale.docstrings.set_docstrings(LinearRegression)
