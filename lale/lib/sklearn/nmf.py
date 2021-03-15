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
from sklearn.decomposition import NMF as SKLModel

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Non-Negative Matrix Factorization (NMF)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "init",
                "solver",
                "beta_loss",
                "tol",
                "max_iter",
                "random_state",
                "alpha",
                "l1_ratio",
                "verbose",
                "shuffle",
            ],
            "relevantToOptimizer": [
                "n_components",
                "tol",
                "max_iter",
                "alpha",
                "shuffle",
            ],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 256,
                            "distribution": "uniform",
                        },
                        {
                            "description": "If not set, keep all components.",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                    "description": "Number of components.",
                },
                "init": {
                    "enum": ["custom", "nndsvd", "nndsvda", "nndsvdar", "random", None],
                    "default": None,
                    "description": "Method used to initialize the procedure.",
                },
                "solver": {
                    "enum": ["cd", "mu"],
                    "default": "cd",
                    "description": "Numerical solver to use:",
                },
                "beta_loss": {
                    "description": "Beta divergence to be minimized, measuring the distance between X and the dot product WH.",
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": -1,
                            "maximumForOptimizer": 1,
                        },
                        {"enum": ["frobenius", "kullback-leibler", "itakura-saito"]},
                    ],
                    "default": "frobenius",
                },
                "tol": {
                    "type": "number",
                    "minimum": 0.0,
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance of the stopping condition.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 200,
                    "description": "Maximum number of iterations before timing out.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Used for initialization and in coordinate descent.",
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0,
                    "description": "Constant that multiplies the regularization terms. Set it to zero to have no regularization.",
                },
                "l1_ratio": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "The regularization mixing parameter.",
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": 0,
                    "description": "Whether to be verbose.",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, randomize the order of coordinates in the CD solver.",
                },
            },
        },
        {
            "description": "beta_loss, only in 'mu' solver",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "beta_loss": {"enum": ["frobenius"]},
                    },
                },
                {
                    "type": "object",
                    "properties": {
                        "solver": {"enum": ["mu"]},
                    },
                },
            ],
        },
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "minimum": 0.0},
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
            "items": {"type": "array", "items": {"type": "number", "minimum": 0.0}},
        }
    },
}

_output_transform_schema = {
    "description": "Transformed data",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Non-negative matrix factorization`_ transformer from scikit-learn for linear dimensionality reduction.

.. _`Non-negative matrix factorization`: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.nmf.html",
    "import_from": "sklearn.decomposition",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

NMF: lale.operators.PlannedIndividualOp
NMF = lale.operators.make_operator(SKLModel, _combined_schemas)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.NMF.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.decomposition.NMF.html
    from lale.schemas import AnyOf, Enum, Null

    NMF = NMF.customize_schema(
        regularization=AnyOf(
            desc="Select whether the regularization affects the components (H), the transformation (W), both or none of them.",
            types=[
                Enum(values=["both", "components", "transformation"]),
                Null(),
            ],
            default="both",
            forOptimizer=True,
        )
    )

lale.docstrings.set_docstrings(NMF)
