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

import lale.docstrings
import lale.helpers
import lale.operators


class RelationalImpl:
    def __init__(self, operator=None):
        self.operator = operator

    def fit(self, X, y=None):
        if self.operator is None:
            raise ValueError("The pipeline object can't be None at the time of fit.")
        if isinstance(X, list):
            raise ValueError(
                """Relational operator's fit does not accept data before join and aggregates.
    Please pass a preprocessed dataset that is either a numpy array or a pandas dataframe."""
            )
        return self

    def transform(self, X):
        if isinstance(X, list):
            raise ValueError(
                """Relational operator's transform does not accept data before join and aggregates.
    Please pass a preprocessed dataset that is either a numpy array or a pandas dataframe."""
            )
        return X


_input_fit_schema = {
    "description": "Input data schema for fit.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            # commenting out to bypass schema validation as of now.
            #   'anyOf': [
            #     { 'type': 'array',
            #       'items': {
            #         'type': 'array',
            #         'items': {'type': 'number'}}}]
            "laleType": "any",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
    },
}

_input_transform_schema = {
    "description": "Input data schema for transform.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            # commenting out to bypass schema validation as of now.
            # 'type': 'array',
            # 'items': {
            #     'type': 'array',
            #     'items': {
            #         'type': 'number'},
            # },
            "laleType": "any",
            "description": "The input data for transform.",
        }
    },
}

_output_transform_schema = {
    "description": "Output data schema for transform.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "operator": {
                    "description": "A lale pipeline object to be used inside of relational that captures the data join and aggregate operations.",
                    "laleType": "operator",
                }
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Higher order operator that contains a nested data join pipeline that has
    multiple table joins and aggregates on those joins.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.relational.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(RelationalImpl, _combined_schemas)

Relational = lale.operators.make_operator(RelationalImpl, _combined_schemas)
