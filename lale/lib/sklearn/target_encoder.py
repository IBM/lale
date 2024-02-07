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

import sklearn.preprocessing
from packaging import version

import lale.docstrings
import lale.operators


class _TargetEncoderNotFoundImpl:
    def __init__(self, **hyperparams):
        raise NotImplementedError(
            "TargetEncoder is only available with scikit-learn versions >= 1.3"
        )

    def transform(self, X):
        raise NotImplementedError(
            "TargetEncoder is only available with scikit-learn versions >= 1.3"
        )


_hyperparams_schema = {
    "description": "Hyperparameter schema for the TargetEncoder model from scikit-learn.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["categories", "target_type"],
            "relevantToOptimizer": [],
            "properties": {
                "categories": {
                    "anyOf": [
                        {
                            "description": "Determine categories automatically from training data.",
                            "enum": ["auto"],
                        },
                        {
                            "description": "The ith list element holds the categories expected in the ith column.",
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "Should be sorted.",
                                    },
                                ]
                            },
                        },
                    ],
                    "default": "auto",
                    "description": "Categories (unique values) per feature.",
                },
                "target_type": {
                    "anyOf": [
                        {
                            "enum": ["auto"],
                            "description": "Type of target is inferred with type_of_target.",
                        },
                        {"enum": ["continuous"], "description": "Continuous target"},
                        {"enum": ["binary"], "description": "Binary target"},
                    ],
                    "description": "Type of target.",
                    "default": "auto",
                },
                "smooth": {
                    "anyOf": [
                        {
                            "enum": ["auto"],
                            "description": "Set to an empirical Bayes estimate.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "A larger smooth value will put more weight on the global target mean",
                        },
                    ],
                    "description": "The amount of mixing of the target mean conditioned on the value of the category with the global target mean.",
                    "default": "auto",
                },
                "cv": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Determines the number of folds in the cross fitting strategy used in fit_transform. For classification targets, StratifiedKFold is used and for continuous targets, KFold is used.",
                    "default": 5,
                },
                "shuffle": {
                    "type": "boolean",
                    "description": "Whether to shuffle the data in fit_transform before splitting into folds. Note that the samples within each split will not be shuffled.",
                    "default": True,
                },
                "random_state": {
                    "description": "When shuffle is True, random_state affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. Pass an int for reproducible output across multiple function calls.",
                    "anyOf": [
                        {
                            "enum": [None],
                        },
                        {
                            "description": "Use the provided random state, only affecting other users of that same random state instance.",
                            "laleType": "numpy.random.RandomState",
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                },
            },
        }
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "y": {
            "description": "The target data used to encode the categories.",
            "type": "array",
        },
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        }
    },
}

_output_transform_schema = {
    "description": "Transformed input; the outer array is over samples.",
    "type": "array",
    "items": {
        "anyOf": [
            {"type": "array", "items": {"type": "number"}},
            {"type": "array", "items": {"type": "string"}},
        ]
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Target encoder`_ for regression and classification targets..

.. _`Target encoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.target_encoder.html",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

if lale.operators.sklearn_version >= version.Version("1.3"):
    TargetEncoder = lale.operators.make_operator(
        sklearn.preprocessing.TargetEncoder, _combined_schemas
    )
else:
    TargetEncoder = lale.operators.make_operator(
        _TargetEncoderNotFoundImpl, _combined_schemas
    )


if lale.operators.sklearn_version >= version.Version("1.4"):
    TargetEncoder = TargetEncoder.customize_schema(
        target_type={
            "anyOf": [
                {
                    "enum": ["auto"],
                    "description": "Type of target is inferred with type_of_target.",
                },
                {"enum": ["continuous"], "description": "Continuous target"},
                {"enum": ["binary"], "description": "Binary target"},
                {"enum": ["multiclass"], "description": "Multiclass target"},
            ],
            "description": "Type of target.",
            "default": "auto",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(TargetEncoder)
