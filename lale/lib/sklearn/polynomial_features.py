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

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Generate polynomial and interaction features.",
    "allOf": [
        {
            "type": "object",
            "required": ["include_bias"],
            "relevantToOptimizer": ["degree", "interaction_only", "include_bias"],
            "additionalProperties": False,
            "properties": {
                "degree": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 3,
                    "default": 2,
                    "description": "The degree of the polynomial features.",
                },
                "interaction_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, only interaction features are produced: features that are products of at most degree distinct input features (so not x[1] ** 2, x[0] * x[2] ** 3, etc.).",
                },
                "include_bias": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True (default), then include a bias column, the feature in which all polynomial powers are zero (i.e. a column of ones - acts as an intercept term in a linear model).",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Compute number of output features.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The data.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "description": "Transform data to polynomial features",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The data to transform, row by row.",
        },
    },
}
_output_transform_schema = {
    "description": "The matrix of features, where NP is the number of polynomial",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Polynomial features`_ transformer from scikit-learn.

.. _`Polynomial features`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.polynomial_features.html",
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

PolynomialFeatures: lale.operators.PlannedIndividualOp
PolynomialFeatures = lale.operators.make_operator(
    sklearn.preprocessing.PolynomialFeatures, _combined_schemas
)

if sklearn.__version__ >= "0.21":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    from lale.schemas import Enum

    PolynomialFeatures = PolynomialFeatures.customize_schema(
        order=Enum(
            values=["C", "F"],
            desc="Order of output array in the dense case. 'F' order is faster to compute, but may slow down subsequent estimators.",
            default="C",
        )
    )


lale.docstrings.set_docstrings(PolynomialFeatures)
