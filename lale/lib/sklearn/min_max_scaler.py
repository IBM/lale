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
import sklearn.preprocessing

import lale.docstrings
import lale.operators

_input_schema_fit = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {},
    },
}

_input_transform_schema = {
    "description": "Input data schema for predictions.",
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

_output_transform_schema = {
    "description": "Output data schema for transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": ["feature_range", "copy"],
            "relevantToOptimizer": [],  # ['feature_range'],
            "properties": {
                "feature_range": {
                    "description": "Desired range of transformed data.",
                    "type": "array",
                    "laleType": "tuple",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": [
                        {
                            "type": "number",
                            "minimumForOptimizer": -1,
                            "maximumForOptimizer": 0,
                        },
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.001,
                            "maximumForOptimizer": 1,
                        },
                    ],
                    "default": [0, 1],
                },
                "copy": {
                    "description": "Set to False to perform inplace row normalization and avoid "
                    "a copy (if the input is already a numpy array).",
                    "type": "boolean",
                    "default": True,
                },
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Min-max scaler`_ transformer from scikit-learn.

.. _`Min-max scaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.min_max_scaler.html",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_schema_fit,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

MinMaxScaler: lale.operators.PlannedIndividualOp
MinMaxScaler = lale.operators.make_operator(
    sklearn.preprocessing.MinMaxScaler, _combined_schemas
)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.22/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    MinMaxScaler = MinMaxScaler.customize_schema(
        clip={
            "type": "boolean",
            "description": "Set to True to clip transformed values of held-out data to provided feature range.",
            "default": False,
        },
    )


lale.docstrings.set_docstrings(MinMaxScaler)
