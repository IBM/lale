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


class _JoinImpl:
    def __init__(self, pred=None, join_limit=None, sliding_window_length=None):
        self._hyperparams = {
            "pred": pred,
            "join_limit": join_limit,
            "sliding_window_length": sliding_window_length,
        }

    def transform(self, X):
        raise NotImplementedError()


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["pred"],
            "relevantToOptimizer": [],
            "properties": {
                "pred": {
                    "description": "Join predicate. Given as Python AST expression.",
                    "laleType": "Any",
                },
                "join_limit": {
                    "description": """For join paths that are one-to-many, join_limit is use to sample the joined results.
When the right hand side of the join has a timestamp column, the join_limit is applied to select the most recent rows.
When the right hand side does not have a timestamp, it randomly samples join_limit number of rows.
Sampling is applied after each pair of tables are joined.""",
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                },
                "sliding_window_length": {
                    "description": """sliding_window_length is also used for sampling the joined results,
only rows in a recent window of length sliding_window_length seconds is used in addition to join_limit.""",
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
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
            "description": "List of tables.",
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
    "description": "Relational algebra join operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.join.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Join = lale.operators.make_operator(_JoinImpl, _combined_schemas)

lale.docstrings.set_docstrings(Join)
