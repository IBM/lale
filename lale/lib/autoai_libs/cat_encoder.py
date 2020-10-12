# Copyright 2020 IBM Corporation
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

import autoai_libs.transformers.exportable
import numpy as np

import lale.docstrings
import lale.operators


class CatEncoderImpl:
    def __init__(
        self,
        encoding,
        categories,
        dtype,
        handle_unknown,
        sklearn_version_family=None,
        activate_flag=True,
        encode_unknown_with="auto",
    ):
        self._hyperparams = {
            "encoding": encoding,
            "categories": categories,
            "dtype": dtype,
            "handle_unknown": handle_unknown,
            "sklearn_version_family": sklearn_version_family,
            "activate_flag": activate_flag,
        }
        self.encode_unknown_with = encode_unknown_with
        self._wrapped_model = autoai_libs.transformers.exportable.CatEncoder(
            **self._hyperparams
        )

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        try:
            return self._wrapped_model.transform(X)
        except ValueError as e:
            if self._wrapped_model.encoding == "ordinal":
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                (transformed_X, X_mask) = self._wrapped_model.encoder._transform(
                    X, handle_unknown="ignore"
                )
                # transformed_X is output with the encoding of the unknown category in column i set to be same
                # as encoding of the first element in categories_[i] and X_mask is a boolean mask
                # that indicates which values were unknown.
                n_features = transformed_X.shape[1]
                for i in range(n_features):
                    if self.encode_unknown_with == "auto":
                        transformed_X[:, i][~X_mask[:, i]] = len(
                            self._wrapped_model.encoder.categories_[i]
                        )
                    else:
                        transformed_X[:, i][~X_mask[:, i]] = self.encode_unknown_with
                    transformed_X[:, i] = transformed_X[:, i].astype(
                        self._wrapped_model.encoder.categories_[i].dtype
                    )
                # Following lines are borrowed from CatEncoder as is:
                if (
                    isinstance(transformed_X[0], np.ndarray)
                    and transformed_X[0].shape[0] == 1
                ):
                    # this is a numpy array whose elements are numpy arrays (arises from string targets)
                    transformed_X = np.concatenate(transformed_X).ravel()
                    if transformed_X.ndim > 1 and transformed_X.shape[1] == 1:
                        transformed_X = transformed_X.reshape(-1, 1)
                transformed_X = transformed_X.reshape(transformed_X.shape[0], -1)
                return transformed_X
            else:
                raise e


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "encoding",
                "categories",
                "dtype",
                "handle_unknown",
                "sklearn_version_family",
                "activate_flag",
                "encode_unknown_with",
            ],
            "relevantToOptimizer": ["encoding"],
            "properties": {
                "encoding": {
                    "description": "The type of encoding to use.",
                    "enum": ["onehot", "ordinal"],
                    "default": "ordinal",
                },
                "categories": {
                    "description": "Categories (unique values) per feature.",
                    "anyOf": [
                        {
                            "description": "Determine categories automatically from training data.",
                            "enum": ["auto", None],
                        },
                        {
                            "description": "The ith list element holds the categories expected in the ith column.",
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "array", "items": {"type": "string"},},
                                    {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "Should be sorted.",
                                    },
                                ]
                            },
                        },
                    ],
                    "default": "auto",
                },
                "dtype": {
                    "description": "Desired dtype of output, must be number. See https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.scalars.html#arrays-scalars-built-in",
                    "laleType": "Any",
                    "default": "float64",
                },
                "handle_unknown": {
                    "description": """Whether to raise an error or ignore if an unknown categorical feature is present during transform.
When this parameter is set to `ignore` and an unknown category is encountered during transform,
the resulting one-hot encoded columns for this feature will be all zeros for encoding 'onehot' and
the resulting encoding with be set to the value indicated by `encode_unknown_with` for encoding 'ordinal'.
In the inverse transform, an unknown category will be denoted as None.""",
                    "enum": ["error", "ignore"],
                    "default": "ignore",
                },
                "sklearn_version_family": {
                    "description": "The sklearn version for backward compatibiity with versions 019 and 020dev. Currently unused.",
                    "enum": ["20", "23", None],
                    "default": None,
                },
                "activate_flag": {
                    "description": "If False, transform(X) outputs the input numpy array X unmodified.",
                    "type": "boolean",
                    "default": True,
                },
                "encode_unknown_with": {
                    "description": """When an unknown categorical feature value is found during transform, and 'handle_unknown' is
set to 'ignore', and encoding is 'ordinal', that value is encoded with this value. Default of 'auto' sets it to an integer equal to n+1, where
n is the maximum encoding value based on known categories.""",
                    "anyOf": [{"type": "integer"}, {"enum": ["auto"]}],
                    "default": "auto",
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        },
        "y": {"laleType": "Any"},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Encoding of categorical features as numbers, currently internally uses the sklearn OneHotEncoder_ and OrdinalEncoder_.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs
.. _OneHotEncoder: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
.. _OrdinalEncoder: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.cat_encoder.html",
    "import_from": "autoai_libs.transformers.exportable",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(CatEncoderImpl, _combined_schemas)

CatEncoder = lale.operators.make_operator(CatEncoderImpl, _combined_schemas)
