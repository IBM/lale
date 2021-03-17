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
            "required": ["strategy", "quantile"],
            "property": {
                "strategy": {
                    "description": """Strategy to use to generate predictions.
- “mean”: always predicts the mean of the training set
- “median”: always predicts the median of the training set
- “quantile”: always predicts a specified quantile of the training set, provided with the quantile parameter.
- “constant”: always predicts a constant value that is provided by the user.""",
                    "enum": ["mean", "median", "quantile", "constant"],
                    "default": "mean",
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
                "quantile": {
                    "description": "The quantile to predict using the “quantile” strategy. A quantile of 0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the maximum.",
                    "type": "float",
                },
            },
        }
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
            "description": "Target values.",
            "type": "array",
            "items": {"type": "number"},
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
    "description": "Predicted values per sample.",
    "type": "array",
    "items": {"type": "number"},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Dummy regressor`_ regressor that makes predictions using simple rules.

.. _`Dummy regressor`: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html
""",
    "import_from": "sklearn.dummy",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

DummyRegressor = lale.operators.make_operator(
    sklearn.dummy.DummyRegressor, _combined_schemas
)

lale.docstrings.set_docstrings(DummyRegressor)
