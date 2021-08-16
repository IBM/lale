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

import lale.docstrings
import lale.operators


class _TeeImpl:
    def __init__(self, listener=None):
        self._listener = listener

    def transform(self, X, y=None):
        if self._listener is not None:
            self._listener(X, y)
        return X

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        return s_X


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "listener": {
                    "anyOf": [
                        {
                            "description": "A callable (lambda, method, class that implements __call__, ...)"
                            "that accepts to arguments: X and y (which may be None).  When transform"
                            "is called on this operator, the callable will be passed the given"
                            "X and y values",
                            "laleType": "callable",
                        },
                        {
                            "description": "No listener.  Causes this operator to behave like NoOp.",
                            "enum": [None],
                        },
                    ]
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
            "description": "Features; no restrictions on data type.",
            "laleType": "Any",
        }
    },
}

_output_transform_schema = {
    "description": "Features; no restrictions on data type.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Passes the data through unchanged (like NoOp), first giving it to an listener.  Useful for debugging and logging."
    "Similar to Observing, which provides a higher order operator with more comprehensive abilities.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.tee.html",
    "import_from": "lale.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Tee = lale.operators.make_operator(_TeeImpl, _combined_schemas)

lale.docstrings.set_docstrings(Tee)
