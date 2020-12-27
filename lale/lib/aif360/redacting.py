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

import pandas as pd

import lale.docstrings
import lale.operators

from .util import _dataset_fairness_properties


def _redaction_value(column_values):
    all_numbers = all([isinstance(val, (int, float)) for val in column_values])
    value_to_count = {}
    all_numbers = True
    for val in column_values:
        value_to_count[val] = value_to_count.get(val, 0) + 1
        if all_numbers and len(value_to_count) > 10:
            break
    if all_numbers and len(value_to_count) > 10:
        result = sum(column_values) / len(column_values)
    else:
        result = None
        for val, count in value_to_count.items():
            if result is None or count > value_to_count[result]:
                result = val
    return result


class RedactingImpl:
    def __init__(self, protected_attribute_names):
        self.protected_attribute_names = protected_attribute_names

    def fit(self, X, y=None):
        self.redaction_values = {
            pa: _redaction_value(X[pa]) for pa in self.protected_attribute_names
        }
        return self

    def transform(self, X):
        new_columns = [
            (
                X[name].map(lambda val: self.redaction_values[name])
                if name in self.redaction_values
                else X[name]
            )
            for name in X.columns
        ]
        result = pd.concat(new_columns, axis=1)
        return result

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        return s_X


_input_fit_schema = {
    "description": "Input data schema for training.",
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
        "y": {"description": "Target values; the array is over samples."},
    },
}

_input_transform_schema = {
    "description": "Input data schema for transform.",
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
    },
}

_output_transform_schema = {
    "description": "Output data schema for transform.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["protected_attribute_names"],
            "relevantToOptimizer": [],
            "properties": {
                "protected_attribute_names": _dataset_fairness_properties[
                    "protected_attribute_names"
                ],
            },
        }
    ],
}

_combined_schemas = {
    "description": """Redacting preprocessor for fairness mitigation.

This sets all the protected attributes to constants. For numbers that
have more than 10 unique values in the column, use the arithmetic mean.
Otherwise, use the most frequent value in the column.
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.redacting.html",
    "import_from": "aif360.sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(RedactingImpl, _combined_schemas)

Redacting = lale.operators.make_operator(RedactingImpl, _combined_schemas)
