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

import sklearn.ensemble

import lale.docstrings
import lale.operators


class VotingClassifierImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.ensemble.VotingClassifier(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)


_hyperparams_schema = {
    "description": "Soft Voting/Majority Rule classifier for unfitted estimators.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimators",
                "voting",
                "weights",
                "n_jobs",
                "flatten_transform",
            ],
            "relevantToOptimizer": ["voting"],
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
                "voting": {
                    "enum": ["hard", "soft"],
                    "default": "hard",
                    "description": "If 'hard', uses predicted class labels for majority rule voting.",
                },
                "weights": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"},},
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
                "flatten_transform": {
                    "type": "boolean",
                    "default": True,
                    "description": "Affects shape of transform output only when voting='soft'",
                },
            },
        },
        {
            "description": "Parameter: flatten_transform > only when voting='soft' if voting='soft' and flatten_transform=true",
            "anyOf": [
                {"type": "object", "properties": {"voting": {"enum": ["soft"]},}},
                {
                    "type": "object",
                    "properties": {"flatten_transform": {"enum": [True]},},
                },
            ],
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
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "Training vectors, where n_samples is the number of samples and",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values.",
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
_input_transform_schema = {
    "description": "Return class labels or probabilities for X for each estimator.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "Training vectors, where n_samples is the number of samples and",
        },
    },
}
_output_transform_schema = {
    "description": "If `voting='soft'` and `flatten_transform=True`:",
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
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "The input samples.",
        },
    },
}
_output_predict_schema = {
    "description": "Predicted class labels.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "description": "Compute probabilities of possible outcomes for samples in X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "The input samples.",
        },
    },
}
_output_predict_proba_schema = {
    "description": "Weighted average probability for each class per sample.",
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
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

VotingClassifier: lale.operators.IndividualOp
VotingClassifier = lale.operators.make_operator(VotingClassifierImpl, _combined_schemas)

if sklearn.__version__ >= "0.21":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.VotingClassifier.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.ensemble.VotingClassifier.html
    from lale.schemas import JSON

    VotingClassifier = VotingClassifier.customize_schema(
        estimators=JSON(
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "laleType": "tuple",
                    "items": [
                        {"type": "string"},
                        {"anyOf": [{"laleType": "operator"}, {"enum": [None, "drop"]}]},
                    ],
                },
                "description": "List of (string, estimator) tuples. Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones.",
            }
        )
    )

lale.docstrings.set_docstrings(VotingClassifierImpl, VotingClassifier._schemas)
