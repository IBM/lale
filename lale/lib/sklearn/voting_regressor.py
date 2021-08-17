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
import sklearn.ensemble

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Prediction voting regressor for unfitted estimators.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimators",
                "weights",
                "n_jobs",
            ],
            "relevantToOptimizer": ["weights"],
            "additionalProperties": False,
            "properties": {
                "estimators": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "laleType": "tuple",
                        "items": [
                            {"type": "string"},
                            {"anyOf": [{"laleType": "operator"}, {"enum": [None]}]},
                        ],
                    },
                    "description": "List of (string, estimator) tuples. Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones.",
                },
                "weights": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Sequence of weights (`float` or `int`) to weight the occurrences of",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of jobs to run in parallel for ``fit``.",
                },
            },
        },
    ],
}
_input_fit_schema = {
    "description": "Fit the estimators.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Input samples.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
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
            "description": "Sample weights. If None, then samples are equally weighted.",
        },
    },
}
_input_fit_transform_schema = {
    "description": "Return class labels or probabilities for X for each estimator. Return predictions for X for each estimator.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Input samples",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "default": "None",
            "description": "Target values. (None for unsupervised transformations.)",
        },
    },
}
_output_fit_transform_schema = {
    "description": "Transformed array.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "number"},
        },
    },
}
_input_transform_schema = {
    "description": "Return predictions for X for each estimator.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Input samples",
        },
    },
}
_output_transform_schema = {
    "description": "Values predicted by each regressor",
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "anyOf": [
                {"type": "number"},
                {"type": "array", "items": {"type": "number"}},
            ]
        },
    },
}

_input_predict_schema = {
    "description": "Predict class labels for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples.",
        },
    },
}
_output_predict_schema = {
    "description": "Predicted class labels.",
    "type": "array",
    "items": {"type": "number"},
}
_input_score_schema = {
    "description": "Return the coefficient of determination R^2 of the prediction.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape (n_samples, n_samples_fitted), where n_samples_fitted is the number of samples used in the fitting for the estimator.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "True values for 'X'.",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "description": "Sample weights. If None, then samples are equally weighted.",
        },
    },
}
_output_score_schema = {
    "description": "R^2 of 'self.predict' wrt 'y'",
    "type": "number",
}


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Voting classifier`_ from scikit-learn for voting ensemble.

.. _`Voting classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.voting_classifier.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_score_schema": _input_score_schema,
        "output_score_schema": _output_score_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_fit_transform": _input_fit_transform_schema,
        "output_fit_transform": _output_fit_transform_schema,
    },
}

VotingRegressor: lale.operators.PlannedIndividualOp
VotingRegressor = lale.operators.make_operator(
    sklearn.ensemble.VotingRegressor, _combined_schemas
)

if sklearn.__version__ >= "0.21":
    # old: N/A (new in this version)
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.ensemble.VotingRegressor.html
    VotingRegressor = VotingRegressor.customize_schema(
        estimators={
            "type": "array",
            "items": {
                "type": "array",
                "laleType": "tuple",
                "items": [
                    {"type": "string"},
                    {"anyOf": [{"laleType": "operator"}, {"enum": [None, "drop"]}]},
                ],
            },
            "description": "List of (string, estimator) tuples. Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones.",
        },
        set_as_available=True,
    )

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.21/modules/generated/sklearn.ensemble.VotingRegressor.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.VotingRegressor.html
    VotingRegressor = VotingRegressor.customize_schema(
        estimators={
            "type": "array",
            "items": {
                "type": "array",
                "laleType": "tuple",
                "items": [
                    {"type": "string"},
                    {"anyOf": [{"laleType": "operator"}, {"enum": ["drop"]}]},
                ],
            },
            "description": "List of (string, estimator) tuples. Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones.",
        },
        set_as_available=True,
    )


lale.docstrings.set_docstrings(VotingRegressor)
