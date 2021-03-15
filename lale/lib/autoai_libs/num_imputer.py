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
import numpy as np

import lale.docstrings
import lale.operators


class _NumImputerImpl:
    def __init__(self, strategy, missing_values, activate_flag=True):
        self._hyperparams = {
            "strategy": strategy,
            "missing_values": missing_values,
            "activate_flag": activate_flag,
        }
        self._wrapped_model = autoai_libs.transformers.exportable.NumImputer(
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
            "required": ["strategy", "missing_values", "activate_flag"],
            "relevantToOptimizer": ["strategy"],
            "properties": {
                "strategy": {
                    "description": "The imputation strategy.",
                    "enum": ["mean", "median", "most_frequent"],
                    "default": "mean",
                },
                "missing_values": {
                    "description": "The placeholder for the missing values. All occurrences of missing_values will be imputed.",
                    "anyOf": [
                        {"laleType": "Any"},
                        {
                            "description": "For missing values encoded as np.nan.",
                            "enum": [np.nan],
                        },
                    ],
                    "default": np.nan,
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
    "description": """Operator from `autoai_libs`_. Missing value imputation for numeric features, currently internally uses the sklearn Imputer_.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs
.. _Imputer: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.Imputer.html#sklearn-preprocessing-imputer""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.num_imputer.html",
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


NumImputer = lale.operators.make_operator(_NumImputerImpl, _combined_schemas)

lale.docstrings.set_docstrings(NumImputer)
