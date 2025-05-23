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

import sklearn.neighbors
from packaging import version

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Hyperparameter schema for the KNeighborsRegressor model from scikit-learn.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "n_neighbors",
                "weights",
                "algorithm",
                "leaf_size",
                "p",
                "metric",
                "metric_params",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "n_neighbors",
                "weights",
                "algorithm",
                "p",
                "metric",
            ],
            "properties": {
                "n_neighbors": {
                    "description": "Number of neighbors to use by default for kneighbors queries.",
                    "type": "integer",
                    "distribution": "uniform",
                    "minimum": 1,
                    "laleMaximum": "X/maxItems",  # number of rows
                    "default": 5,
                    "maximumForOptimizer": 100,
                },
                "weights": {
                    "description": "Weight function used in prediction.",
                    "enum": ["uniform", "distance"],
                    "default": "uniform",
                },
                "algorithm": {
                    "description": "Algorithm used to compute the nearest neighbors.",
                    "enum": ["ball_tree", "kd_tree", "brute", "auto"],
                    "default": "auto",
                },
                "leaf_size": {
                    "description": "Leaf size passed to BallTree or KDTree.",
                    "type": "integer",
                    "distribution": "uniform",
                    "minimum": 1,
                    "default": 30,
                    "maximumForOptimizer": 100,
                },
                "p": {
                    "description": "Power parameter for the Minkowski metric.",
                    "type": "integer",
                    "distribution": "uniform",
                    "minimum": 1,
                    "default": 2,
                    "maximumForOptimizer": 3,
                },
                "metric": {
                    "description": "The distance metric to use for the tree.",
                    "enum": ["euclidean", "manhattan", "minkowski"],
                    "default": "minkowski",
                },
                "metric_params": {
                    "description": "Additional keyword arguments for the metric function.",
                    "anyOf": [
                        {"enum": [None]},
                        {
                            "type": "object",
                            "propertyNames": {"pattern": "[_a-zA-Z][_a-zA-Z0-9]*"},
                        },
                    ],
                    "default": None,
                },
                "n_jobs": {
                    "description": "Number of parallel jobs to run for the neighbor search.",
                    "anyOf": [
                        {
                            "description": "1 unless in joblib.parallel_backend context.",
                            "enum": [None],
                        },
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of CPU cores.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": None,
                },
            },
        },
    ],
}

_input_fit_schema = {
    "description": "Input data schema for training the KNeighborsRegressor model from scikit-learn.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
        },
    },
}

_input_predict_schema = {
    "description": "Input data schema for predictions using the KNeighborsRegressor model from scikit-learn.",
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

_output_predict_schema = {
    "description": "Returns predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`K nearest neighbors regressor`_ from scikit-learn.

.. _`K nearest neighbors regressor`: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.k_neighbors_regressor.html",
    "import_from": "sklearn.neighbors",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["estimator", "regressor", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

KNeighborsRegressor = lale.operators.make_operator(
    sklearn.neighbors.KNeighborsRegressor, _combined_schemas
)

if lale.operators.sklearn_version >= version.Version("1.4"):

    KNeighborsRegressor = KNeighborsRegressor.customize_schema(
        metric={
            "anyOf": [
                {
                    "enum": [
                        "cityblock",
                        "cosine",
                        "euclidean",
                        "haversine",
                        "l1",
                        "l2",
                        "manhattan",
                        "precomputed",
                    ],
                },
                {
                    "laleType": "callable",
                    "forOptimizer": False,
                    "description": "It takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors. This works for Scipy’s metrics, but is less efficient than passing the metric name as a string.",
                },
                {
                    "laleType": "Any",
                    "description": "It will be passed directly to the underlying computation routines.",
                    "forOptimizer": False,
                },
            ],
            "description": "The distance metric to use for the tree.",
            "default": "minkowski",
        },
        set_as_available=True,
    )


lale.docstrings.set_docstrings(KNeighborsRegressor)
