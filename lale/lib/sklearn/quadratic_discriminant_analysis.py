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

import sklearn.discriminant_analysis

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Quadratic Discriminant Analysis",
    "allOf": [
        {
            "type": "object",
            "required": ["priors", "store_covariance"],
            "relevantToOptimizer": ["reg_param", "tol"],
            "additionalProperties": False,
            "properties": {
                "priors": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Priors on classes",
                },
                "reg_param": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 0.0,
                    "description": "Regularizes the covariance estimate as",
                },
                "store_covariance": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True the covariance matrices are computed and stored in the",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Threshold used for rank estimation.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Fit the model according to the given training data and parameters.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Training vector, where n_samples is the number of samples and",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "Target values (integers)",
        },
    },
}
_input_predict_schema = {
    "description": "Perform classification on an array of test vectors X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    },
}
_output_predict_schema = {
    "description": "Perform classification on an array of test vectors X.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Return posterior probabilities of classification.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Array of samples/test vectors.",
        },
    },
}
_output_predict_proba_schema = {
    "description": "Posterior probabilities of classification per class.",
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
    "description": """`Quadratic discriminant analysis`_ classifier with a quadratic decision boundary from scikit-learn.

.. _`Quadratic discriminant analysis`: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.quadratic_discriminant_analysis.html",
    "import_from": "sklearn.discriminant_analysis",
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


QuadraticDiscriminantAnalysis = lale.operators.make_operator(
    sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, _combined_schemas
)

lale.docstrings.set_docstrings(QuadraticDiscriminantAnalysis)
