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

import ast

import lale.docstrings
import lale.operators
from lale.datasets.data_schemas import get_table_name


class _ScanImpl:
    def __init__(self, table=None):
        assert table is not None
        if isinstance(table._expr, ast.Attribute):
            self.table_name = table._expr.attr
        else:
            self.table_name = table._expr.slice.value.s

    @classmethod
    def validate_hyperparams(cls, table=None, X=None, **hyperparams):
        valid = isinstance(table._expr, (ast.Attribute, ast.Subscript))
        if valid:
            base = table._expr.value
            valid = isinstance(base, ast.Name) and base.id == "it"
        if valid and isinstance(table._expr, ast.Subscript):
            sub = table._expr.slice
            valid = isinstance(sub, ast.Index) and isinstance(sub.value, ast.Str)
        if not valid:
            raise ValueError("expected `it.table_name` or `it['table name']`")

    def transform(self, X):
        named_datasets = {get_table_name(d): d for d in X}
        if self.table_name in named_datasets:
            return named_datasets[self.table_name]
        raise ValueError(
            f"could not find '{self.table_name}' in {list(named_datasets.keys())}"
        )

    def viz_label(self) -> str:
        return "Scan:\n" + self.table_name


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
                    "description": "Which table to scan.",
                    "laleType": "expression",
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
            "description": "Outermost array dimension is over datasets that have table names.",
            "type": "array",
            "items": {
                "description": "Middle array dimension is over samples (aka rows).",
                "type": "array",
                "items": {
                    "description": "Innermost array dimension is over features (aka columns).",
                    "type": "array",
                    "items": {"laleType": "Any"},
                },
            },
            "minItems": 1,
        }
    },
}

_output_transform_schema = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"laleType": "Any"},
    },
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


Scan = lale.operators.make_operator(_ScanImpl, _combined_schemas)

lale.docstrings.set_docstrings(Scan)
