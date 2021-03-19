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
import sklearn.dummy

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "required": [
                "strategy",
                "random_state",
            ],
            "properties": {
                "strategy": {
                    "description": """Strategy to use to generate predictions.
- “stratified”: generates predictions by respecting the training set’s class distribution.
- “most_frequent”: always predicts the most frequent label in the training set.
- “prior”: always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior.
- “uniform”: generates predictions uniformly at random.
- “constant”: always predicts a constant label that is provided by the user. This is useful for metrics that evaluate a non-majority class""",
                    "enum": [
                        "stratified",
                        "most_frequent",
                        "prior",
                        "uniform",
                        "constant",
                    ],
                    "default": "prior",
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator for shuffling data when solver == ‘sag’, ‘saga’ or ‘liblinear’.",
                    "anyOf": [
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {
                            "description": "Use the provided random state, only affecting other users of that same random state instance.",
                            "laleType": "numpy.random.RandomState",
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                },
                "constant": {
                    "description": "The explicit constant as predicted by the “constant” strategy. This parameter is useful only for the “constant” strategy.",
                    "anyOf": [
                        {"type": ["integer", "string"]},
                        {"enum": [None]},
                        {"default": None},
                    ],
                },
            },
        },
    ]
}

_input_fit_schema = {
    "required": ["X", "y"],
    "type": "object",
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array"},
        },
        "y": {
            "description": "Target class labels.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        }
    },
}

_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Dummy classifier`_ classifier that makes predictions using simple rules.

.. _`Dummy classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
""",
    "import_from": "sklearn.dummy",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


DummyClassifier = lale.operators.make_operator(
    sklearn.dummy.DummyClassifier, _combined_schemas
)

lale.docstrings.set_docstrings(DummyClassifier)
