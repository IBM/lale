# Copyright 2020, 2021 IBM Corporation
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

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators


class _AliasImpl:
    def __init__(self, name=None):
        self.name = name

    @classmethod
    def validate_hyperparams(cls, name=None, **hyperparams):
        if name is None or not name.strip():
            raise ValueError("Alias hyperparam 'name' cannot be None or empty.")

    def transform(self, X):
        return lale.datasets.data_schemas.add_table_name(X, self.name)

    def viz_label(self) -> str:
        return "Alias:\n" + str(self.name)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["name"],
            "relevantToOptimizer": [],
            "properties": {
                "name": {
                    "description": "The table name to be given to the output dataframe.",
                    "type": "string",
                    "pattern": "[^ ]",
                },
            },
        }
    ]
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input table or dataframe",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
            "minItems": 1,
        }
    },
}

_output_transform_schema = {
    "description": "Features; no restrictions on data type.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra alias operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.alias.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Alias = lale.operators.make_operator(_AliasImpl, _combined_schemas)

lale.docstrings.set_docstrings(Alias)
