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

import sklearn
import sklearn.preprocessing
from packaging import version

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "func",
                "inverse_func",
                "validate",
                "accept_sparse",
                "pass_y",
                "check_inverse",
                "kw_args",
                "inv_kw_args",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "func": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "The callable to use for the transformation.",
                },
                "inverse_func": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "The callable to use for the inverse transformation.",
                },
                "validate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Indicate that the input X array should be checked before calling ``func``.",
                },
                "accept_sparse": {
                    "type": "boolean",
                    "default": False,
                    "description": "Indicate that func accepts a sparse matrix as input.",
                },
                "pass_y": {
                    "anyOf": [{"type": "boolean"}, {"enum": ["deprecated"]}],
                    "default": "deprecated",
                    "description": "Indicate that transform should forward the y argument to the inner callable.",
                },
                "check_inverse": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to check that ``func`` followed by ``inverse_func`` leads to the original inputs.",
                },
                "kw_args": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Dictionary of additional keyword arguments to pass to func.",
                },
                "inv_kw_args": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Dictionary of additional keyword arguments to pass to inverse_func.",
                },
            },
        },
        {
            "description": "If validate is False, then accept_sparse has no effect. Otherwise, if accept_sparse is false, sparse matrix inputs will cause an exception to be raised.",
            "anyOf": [
                {"type": "object", "properties": {"validate": {"enum": [False]}}},
                {"type": "object", "laleNot": "X/isSparse"},
                {
                    "type": "object",
                    "properties": {"accept_sparse": {"enum": [True]}},
                },
            ],
        },
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {"laleType": "Any"},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_transform_schema = {"type": "array", "items": {"laleType": "Any"}}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """FunctionTransformer_ from scikit-learn constructs a transformer from an arbitrary callable that operates at the level of an entire dataset.

.. _FunctionTransformer: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.FunctionTransformer.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.function_transformer.html",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

FunctionTransformer: lale.operators.PlannedIndividualOp
FunctionTransformer = lale.operators.make_operator(
    sklearn.preprocessing.FunctionTransformer, _combined_schemas
)

if lale.operators.sklearn_version >= version.Version("0.22"):
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    from lale.schemas import Bool

    FunctionTransformer = FunctionTransformer.customize_schema(
        validate=Bool(
            desc="Indicate that the input X array should be checked before calling ``func``.",
            default=False,
        ),
        pass_y=None,
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("1.1"):
    # old: https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    # new: https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    FunctionTransformer = FunctionTransformer.customize_schema(
        feature_names_out={
            "anyOf": [{"laleType": "callable"}, {"enum": ["one-to-one", None]}],
            "default": None,
            "description": "Determines the list of feature names that will be returned by the ``get_feature_names_out`` method. If it is ‘one-to-one’, then the output feature names will be equal to the input feature names. If it is a callable, then it must take two positional arguments: this ``FunctionTransformer`` (``self``) and an array-like of input feature names (``input_features``). It must return an array-like of output feature names. The ``get_feature_names_out`` method is only defined if ``feature_names_out`` is not None.",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(FunctionTransformer)
