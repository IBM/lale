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
from sklearn.feature_selection import RFE as SKLModel

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Feature ranking with recursive feature elimination.",
    "allOf": [
        {
            "type": "object",
            "required": ["estimator", "n_features_to_select", "step", "verbose"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "estimator": {
                    "description": "A supervised learning estimator with a fit method that provides information about feature importance either through a coef_ attribute or through a feature_importances_ attribute.",
                    "laleType": "operator",
                },
                "n_features_to_select": {
                    "description": "The number of features to select. If None, half of the features are selected.",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "step": {
                    "description": "If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "forOptimizer": False,
                        },
                        {
                            "type": "number",
                            "minimum": 0,
                            "exclusiveMinimum": True,
                            "maximum": 1,
                            "exclusiveMaximum": True,
                            "maximumForOptimizer": 0.5,
                        },
                    ],
                    "default": 1,
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": 0,
                    "description": "Controls verbosity of output.",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit the model to data matrix X and target(s) y.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "type": "array",
            "items": {"type": "number"},
        },
    },
}

_input_transform_schema = {
    "description": "Reduce X to the selected features.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples.",
        }
    },
}

_output_transform_schema = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Recursive feature elimination`_ transformer from scikit-learn.

.. _`Recursive feature elimination`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.rfe.html",
    "import_from": "sklearn.feature_selection",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

RFE: lale.operators.PlannedIndividualOp
RFE = lale.operators.make_operator(SKLModel, _combined_schemas)


if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.feature_selection.RFE.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.feature_selection.RFE.html
    RFE = RFE.customize_schema(
        n_features_to_select={
            "anyOf": [
                {
                    "description": "Half of the features are selected.",
                    "enum": [None],
                },
                {
                    "description": "Absolute number of features to select.",
                    "type": "integer",
                    "minimum": 1,
                    "laleMaximum": "X/items/maxItems",  # number of columns
                    "forOptimizer": False,
                },
                {
                    "description": "Fraction of features to select",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximum": 1.0,
                    "exclusiveMaximum": True,
                },
            ],
            "default": None,
        },
        importance_getter={
            "anyOf": [
                {
                    "description": "Use the feature importance either through a coef_ or feature_importances_ attributes of estimator.",
                    "enum": ["auto"],
                },
                {
                    "description": "Attribute name/path for extracting feature importance (implemented with attrgetter).",
                    "type": "string",
                },
                {
                    "description": "The callable is passed with the fitted estimator and it should return importance for each feature.",
                    "laleType": "callable",
                },
            ],
            "default": "auto",
        },
    )


lale.docstrings.set_docstrings(RFE)
