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


class _NumpyReplaceUnknownValuesImpl:
    def __init__(
        self,
        known_values_list=None,
        filling_values=None,
        missing_values_reference_list=None,
        filling_values_list=None,
    ):
        self._hyperparams = {
            "known_values_list": known_values_list,
            "filling_values": filling_values,
            "missing_values_reference_list": missing_values_reference_list,
            "filling_values_list": filling_values_list,
        }
        self._wrapped_model = (
            autoai_libs.transformers.exportable.NumpyReplaceUnknownValues(
                **self._hyperparams
            )
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
                "known_values_list",
                "filling_values",
                "missing_values_reference_list",
                "filling_values_list",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "known_values_list": {
                    "description": "Reference list of lists of known values for each column.",
                    "anyOf": [
                        {"type": "array", "items": {"laleType": "Any"}},
                        {"enum": [None]},
                    ],
                    "transient": True,
                    "default": None,
                },
                "filling_values": {
                    "description": "Special value assigned to unknown values.",
                    "laleType": "Any",
                    "default": None,
                },
                "missing_values_reference_list": {
                    "description": "Reference list of missing values.",
                    "anyOf": [
                        {"type": "array", "items": {"laleType": "Any"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "filling_values_list": {
                    "description": "list of special value assigned to unknown values.",
                    "anyOf": [
                        {"type": "array", "items": {"laleType": "Any"}},
                        {"enum": [None]},
                    ],
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
    "description": """Operator from `autoai_libs`_. Given a numpy array and a reference list of known values for each column, replaces values that are not part of a reference list with a special value (typically np.nan). This is typically used to remove labels for columns in a test dataset that have not been seen in the corresponding columns of the training dataset.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.numpy_replace_unknown_values.html",
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


NumpyReplaceUnknownValues = lale.operators.make_operator(
    _NumpyReplaceUnknownValuesImpl, _combined_schemas
)

lale.docstrings.set_docstrings(NumpyReplaceUnknownValues)
