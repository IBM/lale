# Copyright 2022 IBM Corporation
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

import warnings

import numpy as np
from packaging import version

try:
    import category_encoders

    catenc_version = version.parse(getattr(category_encoders, "__version__"))

except ImportError:
    catenc_version = None

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "verbose",
                "cols",
                "drop_invariant",
                "return_df",
                "handle_missing",
                "handle_unknown",
                "min_samples_leaf",
                "smoothing",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "verbose": {
                    "type": "integer",
                    "description": "Verbosity of the output, 0 for none.",
                    "default": 0,
                },
                "cols": {
                    "description": "Columns to encode.",
                    "anyOf": [
                        {
                            "enum": [None],
                            "description": "All string columns will be encoded.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    ],
                    "default": None,
                },
                "drop_invariant": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to drop columns with 0 variance.",
                },
                "return_df": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).",
                },
                "handle_missing": {
                    "enum": ["error", "return_nan", "value"],
                    "default": "value",
                    "description": "Given 'value', return the target mean.",
                },
                "handle_unknown": {
                    "enum": ["error", "return_nan", "value"],
                    "default": "value",
                    "description": "Given 'value', return the target mean.",
                },
                "min_samples_leaf": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 1,
                    "maximumForOptimizer": 10,
                    "description": "For regularization the weighted average between category mean and global mean is taken. The weight is an S-shaped curve between 0 and 1 with the number of samples for a category on the x-axis. The curve reaches 0.5 at min_samples_leaf. (parameter k in the original paper)",
                },
                "smoothing": {
                    "type": "number",
                    "default": 1.0,
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximumForOptimizer": 10.0,
                    "description": "Smoothing effect to balance categorical average vs prior. Higher value means stronger regularization. The value must be strictly bigger than 0. Higher values mean a flatter S-curve (see min_samples_leaf).",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_transform_schema = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Target encoder`_ transformer from scikit-learn contrib that encodes categorical features as numbers.

.. _`Target encoder`: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.category_encoders.target_encoder.html",
    "import_from": "category_encoders.target_encoder",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


class _TargetEncoderImpl:
    def __init__(self, **hyperparams):
        if catenc_version is None:
            raise ValueError("The package 'category_encoders' is not installed.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._wrapped_model = category_encoders.TargetEncoder(**hyperparams)

    def fit(self, X, y):
        if catenc_version is None:
            raise ValueError("The package 'category_encoders' is not installed.")
        if np.issubdtype(y.dtype, np.number):
            numeric_y = y
        else:
            from sklearn.preprocessing import LabelEncoder

            trainable_le = LabelEncoder()
            trained_le = trainable_le.fit(y)
            numeric_y = trained_le.transform(y)
        self._wrapped_model.fit(X, numeric_y)
        return self

    def transform(self, X):
        if catenc_version is None:
            raise ValueError("The package 'category_encoders' is not installed.")
        result = self._wrapped_model.transform(X)
        return result


TargetEncoder = lale.operators.make_operator(_TargetEncoderImpl, _combined_schemas)

if catenc_version is not None and catenc_version >= version.Version("2.5.1"):
    TargetEncoder = TargetEncoder.customize_schema(
        hierarchy={
            "laleType": "Any",
            "default": None,
            "description": "A dictionary or a dataframe to define the hierarchy for mapping.",
        },
    )

lale.docstrings.set_docstrings(TargetEncoder)
