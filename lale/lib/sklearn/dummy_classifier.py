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

from ._common_schemas import schema_1D_cats, schema_2D_numbers

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
                    "description": "Strategy to use to generate predictions.",
                    "anyOf": [
                        {
                            "enum": ["stratified"],
                            "description": "Generates predictions by respecting the training set's class distribution.",
                        },
                        {
                            "enum": ["most_frequent"],
                            "description": "Always predicts the most frequent label in the training set.",
                        },
                        {
                            "enum": ["prior"],
                            "description": "Always predicts the class that maximizes the class prior (like 'most_frequent') and predict_proba returns the class prior.",
                        },
                        {
                            "enum": ["uniform"],
                            "description": "Generates predictions uniformly at random.",
                        },
                        {
                            "enum": ["constant"],
                            "description": "Always predicts a constant label that is provided by the user. This is useful for metrics that evaluate a non-majority class",
                            "forOptimizer": False,
                        },
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
                    ],
                    "default": None,
                },
            },
        },
        {
            "description": "The constant strategy requires a non-None value for the constant hyperparameter.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"strategy": {"not": {"enum": ["constant"]}}},
                },
                {
                    "type": "object",
                    "properties": {"constant": {"not": {"enum": [None]}}},
                },
            ],
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
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        "y": {
            "description": "Target class labels.",
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        }
    },
}

_input_predict_proba_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        }
    },
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
        "output_predict": schema_1D_cats,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": schema_2D_numbers,
    },
}

DummyClassifier = lale.operators.make_operator(
    sklearn.dummy.DummyClassifier, _combined_schemas
)

lale.docstrings.set_docstrings(DummyClassifier)
