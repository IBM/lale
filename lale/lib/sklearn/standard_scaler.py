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

import sklearn.preprocessing

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Standardize features by removing the mean and scaling to unit variance",
    "allOf": [
        {
            "type": "object",
            "required": ["copy", "with_mean", "with_std"],
            "relevantToOptimizer": ["with_mean", "with_std"],
            "additionalProperties": False,
            "properties": {
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "If False, try to avoid a copy and do inplace scaling instead.",
                },
                "with_mean": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, center the data before scaling.",
                },
                "with_std": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, scale the data to unit variance (or equivalently, unit standard deviation).",
                },
            },
        },
        {
            "description": "Setting `with_mean` to True does not work on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.",
            "anyOf": [
                {"type": "object", "properties": {"with_mean": {"enum": [False]}}},
                {"type": "object", "laleNot": "X/isSparse"},
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Compute the mean and std to be used for later scaling.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The data used to compute the mean and standard deviation",
        },
        "y": {"description": "Ignored"},
    },
}

_input_transform_schema = {
    "description": "Perform standardization by centering and scaling",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The data used to scale along the features axis.",
        },
        "copy": {
            "anyOf": [{"type": "boolean"}, {"enum": [None]}],
            "default": None,
            "description": "Copy the input X or not.",
        },
    },
}

_output_transform_schema = {
    "description": "Perform standardization by centering and scaling",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Standard scaler`_ transformer from scikit-learn.

.. _`Standard scaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.standard_scaler.html",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


StandardScaler = lale.operators.make_operator(
    sklearn.preprocessing.StandardScaler, _combined_schemas
)

lale.docstrings.set_docstrings(StandardScaler)
