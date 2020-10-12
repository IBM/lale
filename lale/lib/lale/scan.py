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


class ScanImpl:
    def __init__(self, table=None):
        self._hyperparams = {"table": table}

    def transform(self, X):
        raise NotImplementedError()

    def viz_label(self) -> str:
        return "Scan:\n" + str(self._hyperparams["table"])


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["table"],
            "relevantToOptimizer": [],
            "properties": {
                "table": {
                    "description": "Which table to scan. Given as Python AST expression.",
                    "laleType": "Any",
                }
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
            "description": "List of table file names.",
            "type": "array",
            "items": {"type": "string"},
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
    "description": "Scans a database table.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.scan.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(ScanImpl, _combined_schemas)

Scan = lale.operators.make_operator(ScanImpl, _combined_schemas)
