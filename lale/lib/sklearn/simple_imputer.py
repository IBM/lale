# Copyright 2019-2023 IBM Corporation
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


from typing import List, Optional

import numpy as np
import pandas as pd
import sklearn
import sklearn.impute
from packaging import version

import lale.docstrings
import lale.operators


class _SimpleImputerImpl:
    def __init__(self, **hyperparams):
        self._wrapped_model = sklearn.impute.SimpleImputer(**hyperparams)
        self._out_names: Optional[List[str]] = None

    def _find_out_names(self, X):
        if self._out_names is None and isinstance(X, pd.DataFrame):
            self._out_names = [
                c for i, c in enumerate(X.columns) if i not in self._all_missing
            ]

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        missing_mask = sklearn.impute.MissingIndicator(
            missing_values=self._wrapped_model.missing_values,
            features="all",
        ).fit_transform(X, y)
        self._all_missing = [i for i in range(X.shape[1]) if np.all(missing_mask[:, i])]
        self._find_out_names(X)
        return self

    def transform(self, X):
        result = self._wrapped_model.transform(X)
        self._find_out_names(X)
        if isinstance(X, pd.DataFrame):
            assert self._out_names is not None
            assert result.shape[1] == len(self._out_names)
            result = pd.DataFrame(data=result, index=X.index, columns=self._out_names)
        elif self._out_names is not None:
            result = pd.DataFrame(data=result, columns=self._out_names)
        return result

    def transform_schema(self, s_X):
        return s_X


_hyperparams_schema = {
    "description": "Imputation transformer for completing missing values.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "missing_values",
                "strategy",
                "fill_value",
                "verbose",
                "copy",
                "add_indicator",
            ],
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
                    "description": "If True, a copy of X will be created.",
                },
                "add_indicator": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, a MissingIndicator transform will stack onto output of the imputer’s transform.",
                },
            },
        },
        {
            "description": "Imputation not possible when missing_values == 0 and input is sparse. Provide a dense array instead.",
            "anyOf": [
                {"type": "object", "laleNot": "X/isSparse"},
                {
                    "type": "object",
                    "properties": {"missing_values": {"not": {"enum": [0]}}},
                },
            ],
        },
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


SimpleImputer = lale.operators.make_operator(_SimpleImputerImpl, _combined_schemas)

if lale.operators.sklearn_version >= version.Version("1.1"):
    # old: https://scikit-learn.org/1.0/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
    # new: https://scikit-learn.org/1.1/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
    SimpleImputer = SimpleImputer.customize_schema(
        verbose={
            "anyOf": [{"type": "integer"}, {"enum": ["deprecated"]}],
            "default": "deprecated",
            "description": "Controls the verbosity of the imputer. Deprecated since version 1.1: The ‘verbose’ parameter was deprecated in version 1.1 and will be removed in 1.3. A warning will always be raised upon the removal of empty columns in the future version.",
        }
    )

if lale.operators.sklearn_version >= version.Version("1.2"):
    # old: https://scikit-learn.org/1.1/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
    # new: https://scikit-learn.org/1.2/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
    SimpleImputer = SimpleImputer.customize_schema(
        keep_empty_features={
            "type": "boolean",
            "default": False,
            "description": """If True, features that consist exclusively of missing values when fit is called
are returned in results when transform is called. The imputed value is always 0 except when strategy="constant"
in which case fill_value will be used instead.""",
        }
    )

lale.docstrings.set_docstrings(SimpleImputer)
