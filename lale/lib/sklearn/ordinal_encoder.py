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

import typing
from typing import Any, Dict

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing

import lale.docstrings
import lale.operators
from lale.schemas import AnyOf, Enum, Int, Null


class _OrdinalEncoderImpl:
    def __init__(self, **hyperparams):
        base_hyperparams = {
            "categories": hyperparams["categories"],
            "dtype": hyperparams["dtype"],
        }
        if sklearn.__version__ >= "0.24.1":
            if hyperparams["handle_unknown"] != "ignore":
                base_hyperparams["handle_unknown"] = hyperparams["handle_unknown"]
                base_hyperparams["unknown_value"] = hyperparams["unknown_value"]

        self._wrapped_model = sklearn.preprocessing.OrdinalEncoder(**base_hyperparams)

        self.handle_unknown = hyperparams.get("handle_unknown", None)
        self.encode_unknown_with = hyperparams.get("encode_unknown_with", None)
        self.unknown_categories_mapping = (
            []
        )  # used during inverse transform to keep track of mapping of unknown categories

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out = self._wrapped_model.get_params(deep=deep)
        out["handle_unknown"] = self.handle_unknown
        out["encode_unknown_with"] = self.encode_unknown_with
        return out

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        if self.handle_unknown == "ignore":
            n_features = len(self._wrapped_model.categories_)
            for i in range(n_features):
                self.unknown_categories_mapping.append({})
        return self

    def transform(self, X):
        if self.handle_unknown != "ignore":
            result = self._wrapped_model.transform(X)
            if isinstance(X, pd.DataFrame):
                result = pd.DataFrame(data=result, index=X.index, columns=X.columns)
            return result
        else:
            try:
                result = self._wrapped_model.transform(X)
                if isinstance(X, pd.DataFrame):
                    result = pd.DataFrame(data=result, index=X.index, columns=X.columns)
                return result
            except ValueError as e:
                if self.handle_unknown == "ignore":
                    (transformed_X, X_mask) = self._wrapped_model._transform(
                        X, handle_unknown="ignore"
                    )
                    # transformed_X is output with the encoding of the unknown category in column i set to be same
                    # as encoding of the first element in categories_[i] and X_mask is a boolean mask
                    # that indicates which values were unknown.
                    n_features = transformed_X.shape[1]
                    for i in range(n_features):
                        dict_categories = self.unknown_categories_mapping[i]
                        if self.encode_unknown_with == "auto":
                            transformed_X[:, i][~X_mask[:, i]] = len(
                                self._wrapped_model.categories_[i]
                            )
                            dict_categories[
                                len(self._wrapped_model.categories_[i])
                            ] = None
                        else:
                            transformed_X[:, i][
                                ~X_mask[:, i]
                            ] = self.encode_unknown_with
                            dict_categories[self.encode_unknown_with] = None
                        self.unknown_categories_mapping[i] = dict_categories
                        transformed_X[:, i] = transformed_X[:, i].astype(
                            self._wrapped_model.categories_[i].dtype
                        )
                    return transformed_X
                else:
                    raise e

    def inverse_transform(self, X):
        if self.handle_unknown != "ignore":
            return self._wrapped_model.inverse_transform(X)
        else:
            try:
                X_tr = self._wrapped_model.inverse_transform(X)
            except IndexError:  # which means the original inverse transform failed during the last step
                n_samples, _ = X.shape
                n_features = len(self._wrapped_model.categories_)
                # dtype=object in order to insert None values
                X_tr = np.empty((n_samples, n_features), dtype=object)

                for i in range(n_features):
                    for j in range(n_samples):
                        label = X[j, i].astype("int64", copy=False)
                        try:
                            X_tr[j, i] = self._wrapped_model.categories_[i][label]
                        except IndexError:
                            X_tr[j, i] = self.unknown_categories_mapping[i][label]
            return X_tr

    # _hyperparams_schema = {
    #     "description": "Hyperparameter schema for the OrdinalEncoder model from scikit-learn.",
    #     "allOf": [
    #         {
    #             "type": "object",
    #             "additionalProperties": False,
    #             "required": ["categories", "dtype"],
    #             "relevantToOptimizer": [],
    #             "properties": {
    #                 "categories": {
    #                     "anyOf": [
    #                         {
    #                             "description": "Determine categories automatically from training data.",
    #                             "enum": ["auto", None],
    #                         },
    #                         {
    #                             "description": "The ith list element holds the categories expected in the ith column.",
    #                             "type": "array",
    #                             "items": {
    #                                 "anyOf": [
    #                                     {
    #                                         "type": "array",
    #                                         "items": {"type": "string"},
    #                                     },
    #                                     {
    #                                         "type": "array",
    #                                         "items": {"type": "number"},
    #                                         "description": "Should be sorted.",
    #                                     },
    #                                 ]
    #                             },
    #                         },
    #                     ],
    #                     "default": "auto",
    #                 },
    #                 "dtype": {
    #                     "description": "Desired dtype of output, must be number. See https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.scalars.html#arrays-scalars-built-in",
    #                     "laleType": "Any",
    #                     "default": "float64",
    #                 },
    #                 "handle_unknown": {
    #                     "description": """When set to ‘error’ an error will be raised in case an unknown categorical feature is present during transform.
    # When set to ‘use_encoded_value’, the encoded value of unknown categories will be set to the value given for the parameter unknown_value.
    # In inverse_transform, an unknown category will be denoted as None.""",
    #                     "enum": ["error", "use_encoded_value"],
    #                     "default": "error",
    #                 },
    #                 "unknown_value": {
    #                     "description": """When the parameter handle_unknown is set to ‘use_encoded_value’, this parameter is required and will set the encoded value of unknown categories.
    # It has to be distinct from the values used to encode any of the categories in fit.
    # """,
    #                     "anyOf": [{"type": "integer"}, {"enum": [None, np.nan]}],
    #                     "default": None,
    #                 },
    #             },
    #         }
    #     ],
    # }


_hyperparams_schema = {
    "description": "Hyperparameter schema for the OrdinalEncoder model from scikit-learn.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["categories", "dtype"],
            "relevantToOptimizer": [],
            "properties": {
                "categories": {
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
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
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
the resulting encoding with be set to the value indicated by `encode_unknown_with`.
In the inverse transform, an unknown category will be denoted as None.""",
                    "enum": ["error", "ignore"],
                    "default": "ignore",
                },
                "encode_unknown_with": {
                    "description": """When an unknown categorical feature value is found during transform, and 'handle_unknown' is
set to 'ignore', that value is encoded with this value. Default of 'auto' sets it to an integer equal to n+1, where
n is the maximum encoding value based on known categories.""",
                    "anyOf": [{"type": "integer"}, {"enum": ["auto"]}],
                    "default": "auto",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Input data schema for training the OrdinalEncoder model from scikit-learn.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "y": {"description": "Target class labels; the array is over samples."},
    },
}

_input_transform_schema = {
    "description": "Input data schema for predictions using the OrdinalEncoder model from scikit-learn.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        }
    },
}

_output_transform_schema = {
    "description": "Output data schema for predictions (projected data) using the OrdinalEncoder model from scikit-learn.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Ordinal encoder`_ transformer from scikit-learn that encodes categorical features as numbers.

.. _`Ordinal encoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ordinal_encoder.html",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

OrdinalEncoder = lale.operators.make_operator(_OrdinalEncoderImpl, _combined_schemas)

if sklearn.__version__ >= "0.24.1":
    OrdinalEncoder = typing.cast(
        lale.operators.PlannedIndividualOp,
        OrdinalEncoder.customize_schema(
            handle_unknown=Enum(
                desc="""When set to ‘error’ an error will be raised in case an unknown categorical feature is present during transform.
When set to ‘use_encoded_value’, the encoded value of unknown categories will be set to the value given for the parameter unknown_value.
In inverse_transform, an unknown category will be denoted as None.
When this parameter is set to `ignore` and an unknown category is encountered during transform,
the resulting encoding with be set to the value indicated by `encode_unknown_with` (this functionality is added by lale).
""",
                values=["error", "ignore", "use_encoded_value"],
                default="error",
            ),
            unknown_value=AnyOf(
                desc="""When the parameter handle_unknown is set to ‘use_encoded_value’, this parameter is required and will set the encoded value of unknown categories.
It has to be distinct from the values used to encode any of the categories in fit.
""",
                default=None,
                types=[Int(), Enum(values=[np.nan]), Null()],
            ),
        ),
    )

lale.docstrings.set_docstrings(OrdinalEncoder)
