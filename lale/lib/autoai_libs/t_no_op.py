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

import autoai_libs.cognito.transforms.transform_utils

import lale.docstrings
import lale.operators


class _TNoOpImpl:
    def __init__(self, fun, name, datatypes, feat_constraints, tgraph=None):
        self._hyperparams = {
            "fun": fun,
            "name": name,
            "datatypes": datatypes,
            "feat_constraints": feat_constraints,
            "tgraph": tgraph,
        }
        self._wrapped_model = autoai_libs.cognito.transforms.transform_utils.TNoOp(
            **self._hyperparams
        )

    def fit(self, X, y=None, **fit_params):
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        result = self._wrapped_model.transform(X)
        return result


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["fun", "name", "datatypes", "feat_constraints", "tgraph"],
            "relevantToOptimizer": [],
            "properties": {
                "fun": {
                    "description": "Function pointer (ignored).",
                    "laleType": "Any",
                    "default": None,
                },
                "name": {
                    "description": "A string name that uniquely identifies this transformer from others.",
                    "anyOf": [{"type": "string"}, {"enum": [None]}],
                    "default": None,
                },
                "datatypes": {
                    "description": "List of datatypes that are valid input (ignored).",
                    "laleType": "Any",
                    "default": None,
                },
                "feat_constraints": {
                    "description": "Constraints that must be satisfied by a column to be considered a valid input to this transform (ignored).",
                    "laleType": "Any",
                    "default": None,
                },
                "tgraph": {
                    "description": "Should be the invoking TGraph() object.",
                    "anyOf": [{"laleType": "Any"}, {"enum": [None]}],
                    "default": None,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {"description": "Features; no restrictions on data type."},
        "y": {"laleType": "Any"},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {"description": "Features; no restrictions on data type."}},
}

_output_transform_schema = {
    "description": "Features; no restrictions on data type.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Passes the data through unchanged.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.t_no_op.html",
    "import_from": "autoai_libs.cognito.transforms.transform_utils",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


TNoOp = lale.operators.make_operator(_TNoOpImpl, _combined_schemas)

lale.docstrings.set_docstrings(TNoOp)
