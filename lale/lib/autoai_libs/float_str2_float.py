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

import autoai_libs.transformers.exportable

import lale.docstrings
import lale.operators


class FloatStr2FloatImpl:
    def __init__(
        self, dtypes_list, missing_values_reference_list=None, activate_flag=True
    ):
        self._hyperparams = {
            "dtypes_list": dtypes_list,
            "missing_values_reference_list": missing_values_reference_list,
            "activate_flag": activate_flag,
        }
        self._wrapped_model = autoai_libs.transformers.exportable.FloatStr2Float(
            **self._hyperparams
        )

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "dtypes_list",
                "missing_values_reference_list",
                "activate_flag",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "dtypes_list": {
                    "description": "Strings that denote the type of each column of the input numpy array X.",
                    "type": "array",
                    "items": {
                        "enum": [
                            "char_str",
                            "int_str",
                            "float_str",
                            "float_num",
                            "float_int_num",
                            "int_num",
                            "boolean",
                            "Unknown",
                            "missing",
                        ]
                    },
                    "default": None,
                },
                "missing_values_reference_list": {
                    "anyOf": [
                        {
                            "description": "Reference list of missing values in the input numpy array X.",
                            "type": "array",
                            "items": {"laleType": "Any"},
                        },
                        {
                            "description": "If None, default to ``['?', '', '-', np.nan]``.",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
                "activate_flag": {
                    "description": "If False, transform(X) outputs the input numpy array X unmodified.",
                    "type": "boolean",
                    "default": True,
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
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        },
        "y": {"laleType": "Any"},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Replaces columns of strings that represent floats (type ``float_str`` in dtypes_list) to columns of floats and replaces their missing values with np.nan.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.float_str2_float.html",
    "import_from": "autoai_libs.transformers.exportable",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(FloatStr2FloatImpl, _combined_schemas)

FloatStr2Float = lale.operators.make_operator(FloatStr2FloatImpl, _combined_schemas)
