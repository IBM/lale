# Copyright 2019 IBM Corporation
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

from numpy import nan
from sklearn.impute import MissingIndicator as SKLModel

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "inherited docstring for MissingIndicator    Binary indicators for missing values.",
    "allOf": [
        {
            "type": "object",
            "required": ["missing_values", "features", "sparse", "error_on_new"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "missing_values": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"enum": [nan]},
                        {"enum": [None]},
                    ],
                    "description": "The placeholder for the missing values.",
                    "default": nan,
                },
                "features": {
                    "enum": ["missing-only", "all"],
                    "default": "missing-only",
                    "description": "Whether the imputer mask should represent all or a subset of features.",
                },
                "sparse": {
                    "anyOf": [{"type": "boolean"}, {"enum": ["auto"]}],
                    "description": "Whether the imputer mask format should be sparse or dense.",
                    "default": "auto",
                },
                "error_on_new": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True (default), transform will raise an error when there are",
                },
            },
        },
        {
            "description": 'error_on_new, only when features="missing-only"',
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "error_on_new": {"enum": [True]},
                    },
                },
                {
                    "type": "object",
                    "properties": {
                        "features": {"enum": ["missing-only"]},
                    },
                },
            ],
        },
    ],
}
_input_fit_schema = {
    "description": "Fit the transformer on X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Input data, where ``n_samples`` is the number of samples and",
        },
    },
}
_input_transform_schema = {
    "description": "Generate missing values indicator for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input data to complete.",
        },
    },
}
_output_transform_schema = {
    "description": "The missing indicator for input data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "boolean"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Missing values indicator`_ transformer from scikit-learn.

.. _`Missing values indicator`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.missing_indicator.html",
    "import_from": "sklearn.impute",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


MissingIndicator = lale.operators.make_operator(SKLModel, _combined_schemas)

lale.docstrings.set_docstrings(MissingIndicator)
