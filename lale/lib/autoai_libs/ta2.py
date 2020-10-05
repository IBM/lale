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


class TA2Impl:
    def __init__(
        self,
        fun,
        name,
        datatypes1,
        feat_constraints1,
        datatypes2,
        feat_constraints2,
        tgraph=None,
        apply_all=True,
        col_names=None,
        col_dtypes=None,
        col_as_json_objects=None,
    ):
        self._hyperparams = {
            "fun": fun,
            "name": name,
            "datatypes1": datatypes1,
            "feat_constraints1": feat_constraints1,
            "datatypes2": datatypes2,
            "feat_constraints2": feat_constraints2,
            "tgraph": tgraph,
            "apply_all": apply_all,
            "col_names": col_names,
            "col_dtypes": col_dtypes,
            "col_as_json_objects": col_as_json_objects,
        }
        self._wrapped_model = autoai_libs.cognito.transforms.transform_utils.TA2(
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
            "required": [
                "fun",
                "name",
                "datatypes1",
                "feat_constraints1",
                "datatypes2",
                "feat_constraints2",
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
                "datatypes1": {
                    "description": "List of datatypes that are valid input to the first argument of the transformer function (`numeric`, `float`, `int`, `integer`).",
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "feat_constraints1": {
                    "description": "All constraints that must be satisfied by a column to be considered a valid first argument to this transform.",
                    "laleType": "Any",
                    "default": None,
                },
                "datatypes2": {
                    "description": "List of datatypes that are valid input to the second argument of the transformer function (numeric, float, int, etc.).",
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "feat_constraints2": {
                    "description": "All constraints that must be satisfied by a column to be considered a valid second argument to this transform.",
                    "laleType": "Any",
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
    "description": """Operator from `autoai_libs`_. Feature transformation for binary stateless functions, such as sum or product.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.ta2.html",
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

lale.docstrings.set_docstrings(TA2Impl, _combined_schemas)

TA2 = lale.operators.make_operator(TA2Impl, _combined_schemas)
