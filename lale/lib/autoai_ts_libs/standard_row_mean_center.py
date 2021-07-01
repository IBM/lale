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

from autoai_ts_libs.sklearn.small_data_standard_row_mean_center_transformers import (  # type: ignore # noqa
    StandardRowMeanCenter as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators


class _StandardRowMeanCenterImpl:
    def __init__(self, add_noise=True, noise_var=1e-5):
        self._hyperparams = {"add_noise": add_noise, "noise_var": noise_var}
        self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self._wrapped_model.transform(X, y)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["add_noise", "noise_var"],
            "relevantToOptimizer": [],
            "properties": {
                "add_noise": {
                    "description": "Whether zero-mean Gaussian noise should be added to the data rows before standardizing. Only affects rows which have zero standard deviation.",
                    "type": "boolean",
                    "default": True,
                },
                "noise_var": {
                    "description": "Variance of the noise to be added. Only affects rows which have zero standard deviation.",
                    "type": "number",
                    "default": 1e-5,
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
    "required": ["X", "y"],
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

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {"type": "array", "items": {"laleType": "Any"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_ts_libs`_.

.. _`autoai_ts_libs`: https://pypi.org/project/autoai-ts-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.standard_row_mean_center.html",
    "import_from": "autoai_ts_libs.sklearn.small_data_standard_row_mean_center_transformers",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

StandardRowMeanCenter = lale.operators.make_operator(
    _StandardRowMeanCenterImpl, _combined_schemas
)
lale.docstrings.set_docstrings(StandardRowMeanCenter)
