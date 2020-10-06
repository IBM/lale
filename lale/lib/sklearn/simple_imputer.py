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


import numpy as np
import sklearn.impute

import lale.docstrings
import lale.operators


class SimpleImputerImpl:
    def __init__(
        self,
        missing_values=None,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
    ):
        self._hyperparams = {
            "missing_values": missing_values,
            "strategy": strategy,
            "fill_value": fill_value,
            "verbose": verbose,
            "copy": copy,
        }
        self._wrapped_model = sklearn.impute.SimpleImputer(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)

    def transform_schema(self, s_X):
        return s_X


_hyperparams_schema = {
    "description": "Imputation transformer for completing missing values.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["missing_values", "strategy", "fill_value", "verbose", "copy"],
            "relevantToOptimizer": ["strategy"],
            "properties": {
                "missing_values": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"enum": [np.nan]},
                        {"enum": [None]},
                    ],
                    "default": np.nan,
                    "description": "The placeholder for the missing values.",
                },
                "strategy": {
                    "anyOf": [
                        {"enum": ["constant"], "forOptimizer": False},
                        {"enum": ["mean", "median", "most_frequent"]},
                    ],
                    "default": "mean",
                    "description": "The imputation strategy.",
                },
                "fill_value": {
                    "anyOf": [{"type": "number"}, {"type": "string"}, {"enum": [None]}],
                    "default": None,
                    "description": 'When strategy == "constant", fill_value is used to replace all occurrences of missing_values',
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Controls the verbosity of the imputer.",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, a copy of X will be created. If False, imputation will",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit the imputer on X.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
            "description": "Input data, where ``n_samples`` is the number of samples and  ``n_features`` is the number of features.",
        },
        "y": {},
    },
}

_input_transform_schema = {
    "description": "Impute all missing values in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
            "description": "The input data to complete.",
        },
    },
}
_output_transform_schema = {
    "description": "The input data to complete.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Simple imputer`_ transformer from scikit-learn for completing missing values.

.. _`Simple imputer`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.simple_imputer.html",
    "import_from": "sklearn.impute",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(SimpleImputerImpl, _combined_schemas)

SimpleImputer = lale.operators.make_operator(SimpleImputerImpl, _combined_schemas)
