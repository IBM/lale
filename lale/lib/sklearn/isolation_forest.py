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
from sklearn.ensemble import IsolationForest as SKLModel

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": """Isolation Forest Algorithm.
Return the anomaly score of each sample using the IsolationForest algorithm.
The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
Since recursive partitioning can be represented by a tree structure,
the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively
produce shorter path lengths for particular samples, they are highly likely to be anomalies.""",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_estimators",
                "max_samples",
                "contamination",
                "max_features",
                "bootstrap",
                "n_jobs",
                "behaviour",
                "random_state",
                "verbose",
                "warm_start",
            ],
            "relevantToOptimizer": [
                "n_estimators",
                "max_samples",
                "max_features",
                "bootstrap",
            ],
            "additionalProperties": False,
            "properties": {
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "The number of base estimators in the ensemble.",
                },
                "max_samples": {
                    "description": "The number of samples to draw from X to train each base estimator.",
                    "anyOf": [
                        {
                            "description": "Draw max_samples samples.",
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/maxItems",  # number of rows
                            "forOptimizer": False,
                        },
                        {
                            "description": "Draw max_samples * X.shape[0] samples.",
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "minimumForOptimizer": 0.2,
                            "maximumForOptimizer": 1.0,
                        },
                        {
                            "description": "Draw max_samples=min(256, n_samples) samples.",
                            "enum": ["auto"],
                        },
                    ],
                    "default": "auto",
                },
                "contamination": {
                    "description": """The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
Used when fitting to define the threshold on the scores of the samples.""",
                    "anyOf": [
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 0.5,
                        },
                        {"enum": ["auto"]},
                    ],
                    "default": "auto",
                },
                "max_features": {
                    "description": "The number of features to draw from X to train each base estimator.",
                    "anyOf": [
                        {
                            "description": "Draw max_features features.",
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                            "forOptimizer": False,
                        },
                        {
                            "description": "Draw max_samples * X.shape[1] features.",
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "minimumForOptimizer": 0.01,
                            "maximumForOptimizer": 1.0,
                        },
                    ],
                    "default": 1.0,
                },
                "bootstrap": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether samples are drawn with (True) or without (False) replacement.",
                },
                "n_jobs": {
                    "description": "The number of jobs to run in parallel for both `fit` and `predict`.",
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
                "behaviour": {
                    "description": "This parameter has no effect, is deprecated, and will be removed.",
                    "enum": ["deprecated"],
                    "default": "deprecated",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": """Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest.
If int, random_state is the seed used by the random number generator""",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Controls the verbosity of the tree building process.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble.",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrices are accepted only if",
        },
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "The target values (class labels in classification, real numbers in",
                },
                {"enum": [None]},
            ]
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "description": "Sample weights. If None, then samples are equally weighted.",
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    },
}

_output_predict_schema = {
    "type": "array",
    "items": {"type": "number"},
}

_input_decision_function_schema = {
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

_output_decision_function_schema = {
    "type": "array",
    "items": {"type": "number"},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Isolation forest`_ from scikit-learn for getting the anomaly score of each sample using the IsolationForest algorithm.

.. _`Isolation forest`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.isolation_forest.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

IsolationForest = lale.operators.make_operator(SKLModel, _combined_schemas)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.22/modules/generated/sklearn.ensemble.IsolationForest.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.IsolationForest.html
    IsolationForest = IsolationForest.customize_schema(
        behaviour=None, set_as_available=True
    )

lale.docstrings.set_docstrings(IsolationForest)
