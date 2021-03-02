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


class _OptStandardScalerImpl:
    def __init__(
        self,
        use_scaler_flag=True,
        num_scaler_copy=True,
        num_scaler_with_mean=True,
        num_scaler_with_std=True,
    ):
        self._hyperparams = {
            "use_scaler_flag": use_scaler_flag,
            "num_scaler_copy": num_scaler_copy,
            "num_scaler_with_mean": num_scaler_with_mean,
            "num_scaler_with_std": num_scaler_with_std,
        }
        self._wrapped_model = autoai_libs.transformers.exportable.OptStandardScaler(
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
                "use_scaler_flag",
                "num_scaler_copy",
                "num_scaler_with_mean",
                "num_scaler_with_std",
            ],
            "relevantToOptimizer": [
                "use_scaler_flag",
                "num_scaler_with_mean",
                "num_scaler_with_std",
            ],
            "properties": {
                "use_scaler_flag": {
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": True,
                    "description": "If False, return the input array unchanged.",
                },
                "num_scaler_copy": {
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": True,
                    "description": "If False, try to avoid a copy and do inplace scaling instead.",
                },
                "num_scaler_with_mean": {
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": True,
                    "description": "If True, center the data before scaling.",
                },
                "num_scaler_with_std": {
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": True,
                    "description": "If True, scale the data to unit variance (or equivalently, unit standard deviation).",
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
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ]
        },
        "y": {},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ]
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Acts like an optional StandardScaler_.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs
.. _StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.opt_standard_scaler.html",
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


OptStandardScaler = lale.operators.make_operator(
    _OptStandardScalerImpl, _combined_schemas
)

lale.docstrings.set_docstrings(OptStandardScaler)
