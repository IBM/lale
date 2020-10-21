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

import sklearn.decomposition

import lale.docstrings
import lale.operators


class PCAImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.decomposition.PCA(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "description": "Hyperparameter schema for the PCA model from scikit-learn.",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.\n",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "n_components",
                "copy",
                "whiten",
                "svd_solver",
                "tol",
                "iterated_power",
                "random_state",
            ],
            "relevantToOptimizer": ["n_components", "whiten", "svd_solver"],
            "properties": {
                "n_components": {
                    "anyOf": [
                        {
                            "description": "If not set, keep all components.",
                            "enum": [None],
                        },
                        {
                            "description": "Use Minka's MLE to guess the dimension.",
                            "enum": ["mle"],
                        },
                        {
                            "description": """Select the number of components such that the amount of variance that needs to be explained is greater than the specified percentage.""",
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "exclusiveMaximum": True,
                        },
                        {
                            "description": "Number of components to keep.",
                            "type": "integer",
                            "minimum": 1,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                            "forOptimizer": False,
                        },
                    ],
                    "default": None,
                },
                "copy": {
                    "description": "If false, overwrite data passed to fit.",
                    "type": "boolean",
                    "default": True,
                },
                "whiten": {
                    "description": """When true, multiply the components vectors by the square root of
n_samples and then divide by the singular values to ensure uncorrelated
outputs with unit component-wise variances.""",
                    "type": "boolean",
                    "default": False,
                },
                "svd_solver": {
                    "description": "Algorithm to use.",
                    "enum": ["auto", "full", "arpack", "randomized"],
                    "default": "auto",
                },
                "tol": {
                    "description": "Tolerance for singular values computed by svd_solver arpack.",
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.0,
                },
                "iterated_power": {
                    "anyOf": [
                        {
                            "description": "Number of iterations for the power method computed by svd_solver randomized.",
                            "type": "integer",
                            "minimum": 0,
                        },
                        {"description": "Pick automatically.", "enum": ["auto"]},
                    ],
                    "default": "auto",
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator for shuffling data.",
                    "anyOf": [
                        {
                            "description": "RandomState used by np.random",
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
        },
        {
            "description": "This class does not support sparse input. See TruncatedSVD for an alternative with sparse data.",
            "type": "object",
            "laleNot": "X/isSparse",
        },
        {
            "description": "Option n_components mle can only be set for svd_solver full or auto.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"n_components": {"not": {"enum": ["mle"]},}},
                },
                {
                    "type": "object",
                    "properties": {"svd_solver": {"enum": ["full", "auto"]},},
                },
            ],
        },
        {
            "description": "Setting 0 < n_components < 1 only works for svd_solver full.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "n_components": {
                            "not": {
                                "type": "number",
                                "minimum": 0.0,
                                "exclusiveMinimum": True,
                                "maximum": 1.0,
                                "exclusiveMaximum": True,
                            },
                        }
                    },
                },
                {"type": "object", "properties": {"svd_solver": {"enum": ["full"]},}},
            ],
        },
        {
            "description": "Option tol can be set for svd_solver arpack.",
            "anyOf": [
                {"type": "object", "properties": {"tol": {"enum": [0.0]}}},
                {"type": "object", "properties": {"svd_solver": {"enum": ["arpack"]},}},
            ],
        },
        {
            "description": "Option iterated_power can be set for svd_solver randomized.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"iterated_power": {"enum": ["auto"]},},
                },
                {
                    "type": "object",
                    "properties": {"svd_solver": {"enum": ["randomized"]},},
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
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target for supervised learning (ignored).",
            "laleType": "Any",
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
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Principal component analysis`_ transformer from scikit-learn for linear dimensionality reduction.

.. _`Principal component analysis`: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html",
    "import_from": "sklearn.decomposition",
    "type": "object",
    "tags": {"pre": ["~categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(PCAImpl, _combined_schemas)

PCA = lale.operators.make_operator(PCAImpl, _combined_schemas)
