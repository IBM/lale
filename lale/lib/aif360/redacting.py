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

import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators

from .util import (
    _categorical_fairness_properties,
    _categorical_input_transform_schema,
    _categorical_output_transform_schema,
    _categorical_unsupervised_input_fit_schema,
)


def _redaction_value(column_values):
    all_numbers = all([isinstance(val, (int, float)) for val in column_values])
    value_to_count = {}
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


class _RedactingImpl:
    def __init__(self, *, favorable_labels, protected_attributes):
        self.prot_attr_names = [pa["feature"] for pa in protected_attributes]

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.redaction_values = {
                pa: _redaction_value(X[pa]) for pa in self.prot_attr_names
            }
        elif isinstance(X, np.ndarray):
            self.redaction_values = {
                pa: _redaction_value(X[:, pa]) for pa in self.prot_attr_names
            }
        else:
            raise TypeError(f"unexpected type {type(X)}")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            new_columns = [
                (
                    X[name].map(lambda val: self.redaction_values[name])
                    if name in self.redaction_values
                    else X[name]
                )
                for name in X.columns
            ]
            result = pd.concat(new_columns, axis=1)
        elif isinstance(X, np.ndarray):
            result = X.copy()
            for column, value in self.redaction_values.items():
                result[:, column].fill(value)
        else:
            raise TypeError(f"unexpected type {type(X)}")
        return result

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        return s_X


_input_fit_schema = _categorical_unsupervised_input_fit_schema
_input_transform_schema = _categorical_input_transform_schema
_output_transform_schema = _categorical_output_transform_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["favorable_labels", "protected_attributes"],
            "relevantToOptimizer": [],
            "properties": {
                "favorable_labels": {
                    "description": "Ignored.",
                    "laleType": "Any",
                },
                "protected_attributes": _categorical_fairness_properties[
                    "protected_attributes"
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
This operator is used internally by various lale.lib.aif360 metrics
and mitigators, so you often do not need to use it directly yourself.
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.redacting.html#lale.lib.aif360.redacting.Redacting",
    "import_from": "lale.lib.aif360",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Redacting = lale.operators.make_operator(_RedactingImpl, _combined_schemas)

lale.docstrings.set_docstrings(Redacting)
