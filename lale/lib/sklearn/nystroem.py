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

import sklearn.kernel_approximation

import lale.docstrings
import lale.operators


class NystroemImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.kernel_approximation.Nystroem(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "description": "Hyperparameter schema for the Nystroem model from scikit-learn.",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kernel",
                "gamma",
                "coef0",
                "degree",
                "n_components",
                "random_state",
            ],
            "relevantToOptimizer": [
                "kernel",
                "gamma",
                "coef0",
                "degree",
                "n_components",
            ],
            "properties": {
                "kernel": {
                    "description": "Kernel map to be approximated.",
                    "anyOf": [
                        {
                            "description": "keys of sklearn.metrics.pairwise.KERNEL_PARAMS",
                            "enum": [
                                "additive_chi2",
                                "chi2",
                                "cosine",
                                "linear",
                                "poly",
                                "polynomial",
                                "rbf",
                                "laplacian",
                                "sigmoid",
                            ],
                        },
                        {"laleType": "callable", "forOptimizer": False},
                    ],
                    "default": "rbf",
                },
                "gamma": {
                    "description": "Gamma parameter.",
                    "anyOf": [
                        {"enum": [None]},
                        {
                            "type": "number",
                            "distribution": "loguniform",
                            "minimumForOptimizer": 3.0517578125e-05,
                            "maximumForOptimizer": 8,
                        },
                    ],
                    "default": None,
                },
                "coef0": {
                    "description": "Zero coefficient.",
                    "anyOf": [
                        {"enum": [None]},
                        {
                            "type": "number",
                            "minimum": (-1),
                            "distribution": "uniform",
                            "maximumForOptimizer": 1,
                        },
                    ],
                    "default": None,
                },
                "degree": {
                    "description": "Degree of the polynomial kernel.",
                    "anyOf": [
                        {"enum": [None]},
                        {
                            "type": "integer",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 5,
                        },
                    ],
                    "default": None,
                },
                "kernel_params": {
                    "description": "Additional parameters (keyword arguments) for kernel "
                    "function passed as callable object.",
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                },
                "n_components": {
                    "description": "Number of features to construct. How many data points will be used to construct the mapping.",
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 256,
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator.",
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Input data schema for training the Nystroem model from scikit-learn.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {"description": "Target class labels; the array is over samples."},
    },
}

_input_transform_schema = {
    "description": "Input data schema for predictions using the Nystroem model from scikit-learn.",
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
    "description": "Output data schema for predictions (projected data) using the Nystroem model from scikit-learn.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Nystroem`_ transformer from scikit-learn.

.. _`Nystroem`: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.nystroem.html",
    "import_from": "sklearn.kernel_approximation",
    "type": "object",
    "tags": {"pre": ["~categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(NystroemImpl, _combined_schemas)

Nystroem = lale.operators.make_operator(NystroemImpl, _combined_schemas)
