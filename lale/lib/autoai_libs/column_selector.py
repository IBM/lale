# Copyright 2021 IBM Corporation
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
import pandas as pd

import lale.docstrings
import lale.operators


class _ColumnSelectorImpl:
    def __init__(self, columns_indices_list=None, activate_flag=True):
        self._hyperparams = {
            "columns_indices_list": columns_indices_list,
            "activate_flag": activate_flag,
        }
        self._wrapped_model = autoai_libs.transformers.exportable.ColumnSelector(
            **self._hyperparams
        )

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["columns_indices_list", "activate_flag"],
            "relevantToOptimizer": [],
            "properties": {
                "columns_indices_list": {
                    "description": "List of indices to select numpy columns or list elements.",
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer", "minimum": 0}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "activate_flag": {
                    "description": "Determines whether transformer is active or not.",
                    "type": "boolean",
                    "default": True,
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
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Selects a subset of columns for a given numpy array or subset of elements of a list.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.column_selector.html",
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


ColumnSelector = lale.operators.make_operator(_ColumnSelectorImpl, _combined_schemas)

lale.docstrings.set_docstrings(ColumnSelector)
