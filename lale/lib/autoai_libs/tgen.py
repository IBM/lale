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


class TGenImpl:
    def __init__(
        self,
        fun,
        name,
        arg_count,
        datatypes_list,
        feat_constraints_list,
        tgraph=None,
        apply_all=True,
        col_names=None,
        col_dtypes=None,
        col_as_json_objects=None,
    ):
        self._hyperparams = {
            "fun": fun,
            "name": name,
            "arg_count": arg_count,
            "datatypes_list": datatypes_list,
            "feat_constraints_list": feat_constraints_list,
            "tgraph": tgraph,
            "apply_all": apply_all,
            "col_names": col_names,
            "col_dtypes": col_dtypes,
            "col_as_json_objects": col_as_json_objects,
        }
        self._wrapped_model = autoai_libs.cognito.transforms.transform_utils.TGen(
            **self._hyperparams
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
                "fun": {
                    "description": "The function pointer.",
                    "laleType": "Any",
                    "default": None,
                },
                "name": {
                    "description": "A string name that uniquely identifies this transformer from others.",
                    "anyOf": [{"type": "string"}, {"enum": [None]}],
                    "default": None,
                },
                "arg_count": {
                    "description": "Number of arguments to the function, e.g., 1 for unary, 2 for binary, and so on.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                },
                "datatypes_list": {
                    "description": "A list of arg_count lists that correspond to the acceptable input data types for each argument.",
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "description": "List of datatypes that are valid input to the corresponding argument (numeric, float, int, etc.).",
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        {"enum": [None]},
                    ],
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
                    "default": None,
                },
                "tgraph": {
                    "description": "Should be the invoking TGraph() object.",
                    "anyOf": [
                        {"laleType": "Any"},
                        {
                            "enum": [None],
                            "description": "Passing None will result in some failure to detect some inefficiencies due to lack of caching.",
                        },
                    ],
                    "default": None,
                },
                "apply_all": {
                    "description": "Only use applyAll = True. It means that the transformer will enumerate all features (or feature sets) that match the specified criteria and apply the provided function to each.",
                    "type": "boolean",
                    "default": True,
                },
                "col_names": {
                    "description": "Names of the feature columns in a list.",
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "col_dtypes": {
                    "description": "List of the datatypes of the feature columns.",
                    "anyOf": [
                        {"type": "array", "items": {"laleType": "Any"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "col_as_json_objects": {
                    "description": "Names of the feature columns in a json dict.",
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
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

lale.docstrings.set_docstrings(TGenImpl, _combined_schemas)

TGen = lale.operators.make_operator(TGenImpl, _combined_schemas)
