# Copyright 2021, 2022 IBM Corporation
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

import lale.docstrings
import lale.operators

from .project import Project


class _SplitXyImpl:
    def __init__(self, label_name="y"):
        self.label_name = label_name
        self._project_X = None
        self._project_y = None

    def _extract_y(self, X):
        if self._project_y is None:
            self._project_y = Project(columns=[self.label_name])
        result = self._project_y.transform(X)
        if isinstance(result, pd.DataFrame):
            result = result.squeeze()
        return result

    def transform(self, X):
        if self._project_X is None:
            self._project_X = Project(drop_columns=[self.label_name])
        return self._project_X.transform(X)

    def transform_X_y(self, X, y):
        return self.transform(X), self._extract_y(X)

    def viz_label(self) -> str:
        return "SplitXy:\n" + self.label_name


_hyperparams_schema = {
    "description": "The SplitXy operator separates the label field/column from the input dataframe X.",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "label_name": {
                    "description": "The name of the label column in the input dataframe X.",
                    "default": "y",
                    "type": "string",
                },
            },
        }
    ],
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
    },
}

_output_transform_schema = {
    "description": "Output data schema for transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"laleType": "Any"}},
}

_input_transform_X_y_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        "y": {
            "description": "Input labels; ignored.",
            "laleType": "Any",
        },
    },
}

_output_transform_X_y_schema = {
    "type": "array",
    "laleType": "tuple",
    "items": [
        {
            "description": "X",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        {
            "description": "y",
            "type": "array",
            "items": {"laleType": "Any"},
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra SplitXy operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.split_xy.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_transform_X_y": _input_transform_X_y_schema,
        "output_transform_X_y": _output_transform_X_y_schema,
    },
}

SplitXy = lale.operators.make_operator(_SplitXyImpl, _combined_schemas)

lale.docstrings.set_docstrings(SplitXy)
