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

import lale.docstrings
import lale.operators


class MapImpl:
    def __init__(self, columns, remainder="passthrough"):
        self._hyperparams = {"columns": columns, "remainder": remainder}

    def transform(self, X):
        raise NotImplementedError()


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "columns": {
                    "description": "Mappings for producing output columns.",
                    "anyOf": [
                        {
                            "description": "Dictionary of output column names and mapping expressions.",
                            "type": "object",
                            "additionalProperties": {"laleType": "expression"},
                        },
                        {
                            "description": "List of mapping expressions. The output column name is determined by a heuristic based on the input column name and the transformation function.",
                            "type": "array",
                            "items": {"laleType": "expression"},
                        },
                    ],
                    "default": [],
                },
                "remainder": {
                    "description": "Transformation for the remaining columns.",
                    "anyOf": [
                        {"enum": ["passthrough", "drop"]},
                        {"description": "Mapping expression.", "laleType": "operator"},
                    ],
                    "default": "passthrough",
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
            "description": "The outer array is over rows.",
            "type": "array",
            "items": {
                "description": "The inner array is over columns.",
                "type": "array",
                "items": {"laleType": "Any"},
            },
        }
    },
}

_output_transform_schema = {
    "description": "The outer array is over rows.",
    "type": "array",
    "items": {
        "description": "The inner array is over columns.",
        "type": "array",
        "items": {"laleType": "Any"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra map operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.map.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(MapImpl, _combined_schemas)

Map = lale.operators.make_operator(MapImpl, _combined_schemas)
