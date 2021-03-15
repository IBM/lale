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

import sklearn.naive_bayes

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Gaussian Naive Bayes (GaussianNB)",
    "allOf": [
        {
            "type": "object",
            "required": ["priors"],
            "relevantToOptimizer": [],
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
                    "description": "Prior probabilities of the classes. If specified the priors are not",
                },
                "var_smoothing": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "default": 1e-09,
                    "description": "Portion of the largest variance of all features that is added to variances for calculation stability.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Fit Gaussian Naive Bayes according to X, y",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Training vectors, where n_samples is the number of samples",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "Target values.",
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
            "description": "Weights applied to individual samples (1. for unweighted).",
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
    "description": "Predicted target values for X",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Return probability estimates for the test vector X.",
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
_output_predict_proba_schema = {
    "description": "Returns the probability of the samples for each class in",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Gaussian Naive Bayes`_ classifier from scikit-learn.

.. _`Gaussian Naive Bayes`: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.gaussian_naive_bayes.html",
    "import_from": "sklearn.naive_bayes",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}


GaussianNB = lale.operators.make_operator(
    sklearn.naive_bayes.GaussianNB, _combined_schemas
)

lale.docstrings.set_docstrings(GaussianNB)
