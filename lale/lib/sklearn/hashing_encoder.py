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

import pandas as pd

try:
    from category_encoders.hashing import HashingEncoder as SkHashingEncoder
except ImportError:

    class SkHashingEncoder:  # type: ignore
        def __init__(self, *args, **kargs):
            raise ValueError("The package 'category_encoders' is not installed.")

        def fit(self, X, y=None):
            raise ValueError("The package 'category_encoders' is not installed.")

        def transform(self, X):
            raise ValueError("The package 'category_encoders' is not installed.")


import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Hyperparameter schema for the HashingEncoder model from scikit-learn contrib.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["n_components", "cols", "hash_method"],
            "relevantToOptimizer": [],
            "properties": {
                "n_components": {
                    "description": "how many bits to use to represent the feature.",
                    "type": "integer",
                    "default": 8,
                },
                "cols": {
                    "description": "a list of columns to encode, if None, all string columns will be encoded.",
                    "anyOf": [
                        {"enum": [None]},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    ],
                    "default": None,
                },
                "hash_method": {
                    "descruption": "which hashing method to use.",
                    "type": "string",
                    "default": "md5",
                },
            },
        }
    ],
}

_input_fit_schema = {
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
        },
        "y": {"description": "Target class labels; the array is over samples."},
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
    "description": "Hash codes.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Hashing encoder`_ transformer from scikit-learn contrib that encodes categorical features as numbers.

.. _`Hashing encoder`: https://contrib.scikit-learn.org/category_encoders/hashing.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.hashing_encoder.html",
    "import_from": "category_encoders.hashing",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


class _HashingEncoderImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = SkHashingEncoder(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        if isinstance(X, pd.DataFrame):
            self._X_columns = X.columns
        return self

    def transform(self, X):
        result = self._wrapped_model.transform(X)
        return result


HashingEncoder = lale.operators.make_operator(_HashingEncoderImpl, _combined_schemas)

lale.docstrings.set_docstrings(HashingEncoder)
