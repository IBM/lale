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

import lale.datasets.data_schemas
import lale.docstrings
import lale.helpers
import lale.operators

from ._common_schemas import (
    _hparam_col_dtypes,
    _hparams_apply_all,
    _hparams_col_as_json_objects,
    _hparams_col_names,
    _hparams_datatype_spec,
    _hparams_fun_pointer,
    _hparams_tgraph,
    _hparams_transformer_name,
)


class _TGenImpl:
    def __init__(self, **hyperparams):
        self._wrapped_model = autoai_libs.cognito.transforms.transform_utils.TGen(
            **hyperparams
        )

    def fit(self, X, y=None):
        stripped_X = lale.datasets.data_schemas.strip_schema(X)
        self._wrapped_model.fit(stripped_X, y)
        return self

    def transform(self, X):
        stripped_X = lale.datasets.data_schemas.strip_schema(X)
        result = self._wrapped_model.transform(stripped_X)
        return result


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "fun",
                "name",
                "arg_count",
                "datatypes_list",
                "feat_constraints_list",
                "tgraph",
                "apply_all",
                "col_names",
                "col_dtypes",
                "col_as_json_objects",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "fun": _hparams_fun_pointer(description="The function pointer."),
                "name": _hparams_transformer_name,
                "arg_count": {
                    "description": "Number of arguments to the function, e.g., 1 for unary, 2 for binary, and so on.",
                    "type": "integer",
                    "minimum": 1,
                    "transient": "alwaysPrint",  # since positional argument
                    "default": 1,
                },
                "datatypes_list": {
                    "description": "A list of arg_count lists that correspond to the acceptable input data types for each argument.",
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "description": "List of datatypes that are valid input to the corresponding argument (numeric, float, int, etc.).",
                                **_hparams_datatype_spec,
                            },
                        },
                        {"enum": [None]},
                    ],
                    "transient": "alwaysPrint",  # since positional argument
                    "default": None,
                },
                "feat_constraints_list": {
                    "description": "A list of arg_count lists that correspond to some constraints that should be imposed on selection of the input features.",
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "description": "List of feature constraints for the corresponding argument.",
                                "type": "array",
                                "items": {"laleType": "Any"},
                            },
                        },
                        {"enum": [None]},
                    ],
                    "transient": "alwaysPrint",  # since positional argument
                    "default": None,
                },
                "tgraph": _hparams_tgraph,
                "apply_all": _hparams_apply_all,
                "col_names": _hparams_col_names,
                "col_dtypes": _hparam_col_dtypes,
                "col_as_json_objects": _hparams_col_as_json_objects,
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        "y": {"laleType": "Any"},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}}
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {"type": "array", "items": {"laleType": "Any"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Feature transformation via a general wrapper that can be used for most functions (may not be most efficient though).

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.ta1.html",
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


TGen = lale.operators.make_operator(_TGenImpl, _combined_schemas)

lale.docstrings.set_docstrings(TGen)
