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
import sklearn.ensemble

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "An extra-trees classifier.",
    "allOf": [
        {
            "type": "object",
            "required": ["class_weight"],
            "relevantToOptimizer": [
                "n_estimators",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "bootstrap",
            ],
            "additionalProperties": False,
            "properties": {
                "n_estimators": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "default": 10,
                    "description": "The number of trees in the forest.",
                },
                "criterion": {
                    "enum": ["gini", "entropy"],
                    "default": "gini",
                    "description": "The function to measure the quality of a split.",
                },
                "max_depth": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 5,
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The maximum depth of the tree. If None, then nodes are expanded until",
                },
                "min_samples_split": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/maxItems",  # number of rows
                            "forOptimizer": False,
                            "description": "Consider min_samples_split as the minimum number.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "minimumForOptimizer": 0.01,
                            "maximumForOptimizer": 0.5,
                            "default": 0.05,
                            "description": "min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.",
                        },
                    ],
                    "default": 2,
                    "description": "The minimum number of samples required to split an internal node.",
                },
                "min_samples_leaf": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "laleMaximum": "X/maxItems",  # number of rows
                            "forOptimizer": False,
                            "description": "Consider min_samples_leaf as the minimum number.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 0.5,
                            "description": "min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.",
                            "default": 0.05,
                        },
                    ],
                    "default": 1,
                    "description": "The minimum number of samples required to be at a leaf node.",
                },
                "min_weight_fraction_leaf": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.5,
                    "default": 0.0,
                    "description": "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.",
                },
                "max_features": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                            "forOptimizer": False,
                            "description": "Consider max_features features at each split.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "distribution": "uniform",
                            "minimumForOptimizer": 0.01,
                            "default": 0.5,
                            "description": "max_features is a fraction and int(max_features * n_features) features are considered at each split.",
                        },
                        {"enum": ["auto", "sqrt", "log2", None]},
                    ],
                    "default": "auto",
                    "description": "The number of features to consider when looking for the best split:",
                },
                "max_leaf_nodes": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 1000,
                        },
                        {
                            "enum": [None],
                            "description": "Unlimited number of leaf nodes.",
                        },
                    ],
                    "default": None,
                    "description": "Grow trees with ``max_leaf_nodes`` in best-first fashion.",
                },
                "min_impurity_decrease": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximumForOptimizer": 10.0,
                    "default": 0.0,
                    "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                },
                "min_impurity_split": {
                    "anyOf": [{"type": "number", "minimum": 0.0}, {"enum": [None]}],
                    "default": None,
                    "description": "Threshold for early stopping in tree growth. A node will split",
                },
                "bootstrap": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.",
                },
                "oob_score": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use out-of-bag samples to estimate the generalization accuracy.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of jobs to run in parallel for both `fit` and `predict`.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator;",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Controls the verbosity when fitting and predicting.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.",
                },
                "class_weight": {
                    "anyOf": [
                        {"type": "object"},  # dict, list of dicts,
                        {"enum": ["balanced", "balanced_subsample", None]},
                    ],
                    "description": "Weights associated with classes in the form ``{class_label: weight}``.",
                    "default": None,
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Build a forest of trees from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                }
            ],
            "description": "The training input samples. Internally, its dtype will be converted",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "The target values (class labels in classification, real numbers in",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "description": "Sample weights. If None, then samples are equally weighted. Splits",
        },
    },
}
_input_predict_schema = {
    "description": "Predict class for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples. Internally, its dtype will be converted to",
        },
    },
}
_output_predict_schema = {
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Predict class probabilities for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples. Internally, its dtype will be converted to",
        },
    },
}
_output_predict_proba_schema = {
    "description": "such arrays if n_outputs > 1.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Extra trees classifier`_ random forest from scikit-learn.

.. _`Extra trees classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.extra_trees_classifier.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}

ExtraTreesClassifier: lale.operators.PlannedIndividualOp
ExtraTreesClassifier = lale.operators.make_operator(
    sklearn.ensemble.ExtraTreesClassifier, _combined_schemas
)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    from lale.schemas import AnyOf, Float, Int, Null

    ExtraTreesClassifier = ExtraTreesClassifier.customize_schema(
        n_estimators=Int(
            desc="The number of trees in the forest.",
            default=100,
            forOptimizer=True,
            minimumForOptimizer=10,
            maximumForOptimizer=100,
        ),
        ccp_alpha=Float(
            desc="Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.",
            default=0.0,
            forOptimizer=False,
            minimum=0.0,
            maximumForOptimizer=0.1,
        ),
        max_samples=AnyOf(
            types=[
                Null(desc="Draw X.shape[0] samples."),
                Int(desc="Draw max_samples samples.", minimum=1),
                Float(
                    desc="Draw max_samples * X.shape[0] samples.",
                    minimum=0.0,
                    exclusiveMinimum=True,
                    maximum=1.0,
                    exclusiveMaximum=True,
                ),
            ],
            desc="If bootstrap is True, the number of samples to draw from X to train each base estimator.",
            default=None,
        ),
    )


lale.docstrings.set_docstrings(ExtraTreesClassifier)
