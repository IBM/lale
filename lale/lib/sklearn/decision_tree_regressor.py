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
import sklearn.tree

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "A decision tree regressor.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
            ],
            "relevantToOptimizer": [
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
            ],
            "additionalProperties": False,
            "properties": {
                "criterion": {
                    "description": "Function to measure the quality of a split.",
                    "anyOf": [
                        {"enum": ["mse", "friedman_mse"]},
                        {"enum": ["mae"], "forOptimizer": False},
                    ],
                    "default": "mse",
                },
                "splitter": {
                    "enum": ["best", "random"],
                    "default": "best",
                    "description": "Strategy to choose the split at each node.",
                },
                "max_depth": {
                    "description": "Maximum depth of the tree.",
                    "default": None,
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 5,
                        },
                        {
                            "enum": [None],
                            "description": "If None, then nodes are expanded until all leaves are pure, or until all leaves contain less than min_samples_split samples.",
                        },
                    ],
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
                            "minimumForOptimizer": 0.01,
                            "default": 0.05,
                            "description": "min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.",
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
                    "description": "Minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.",
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
                            "minimumForOptimizer": 0.01,
                            "default": 0.5,
                            "distribution": "uniform",
                            "description": "max_features is a fraction and int(max_features * n_features) features are considered at each split.",
                        },
                        {"enum": ["auto", "sqrt", "log2", None]},
                    ],
                    "default": "auto",
                    "description": "The number of features to consider when looking for the best split.",
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator.",
                    "anyOf": [
                        {"laleType": "numpy.random.RandomState"},
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
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
                    "description": "Grow a tree with ``max_leaf_nodes`` in best-first fashion.",
                },
                "min_impurity_decrease": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximumForOptimizer": 10.0,
                    "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                },
                "min_impurity_split": {
                    "anyOf": [{"type": "number", "minimum": 0.0}, {"enum": [None]}],
                    "default": None,
                    "description": "Threshold for early stopping in tree growth.",
                },
                "presort": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to presort the data to speed up the finding of best splits in fitting.",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The target values (real numbers).",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "Samples are equally weighted."},
            ],
            "description": "Sample weights.",
        },
        "check_input": {
            "type": "boolean",
            "default": True,
            "description": "Allow to bypass several input checking.",
        },
        "X_idx_sorted": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "The indexes of the sorted training input samples. If many tree",
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "check_input": {
            "type": "boolean",
            "default": True,
            "description": "Allow to bypass several input checking.",
        },
    },
}

_output_predict_schema = {
    "description": "The predicted classes, or the predict values.",
    "type": "array",
    "items": {"type": "number"},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Decision tree regressor`_ from scikit-learn.

.. _`Decision tree regressor`: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.decision_tree_regressor.html",
    "import_from": "sklearn.tree",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


DecisionTreeRegressor: lale.operators.PlannedIndividualOp
DecisionTreeRegressor = lale.operators.make_operator(
    sklearn.tree.DecisionTreeRegressor, _combined_schemas
)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # new: https://scikit-learn.org/0.22/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    from lale.schemas import AnyOf, Bool, Enum, Float

    DecisionTreeRegressor = DecisionTreeRegressor.customize_schema(
        presort=AnyOf(
            types=[Bool(), Enum(["deprecated"])],
            desc="This parameter is deprecated and will be removed in v0.24.",
            default="deprecated",
        ),
        ccp_alpha=Float(
            desc="Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.",
            default=0.0,
            forOptimizer=False,
            minimum=0.0,
            maximumForOptimizer=0.1,
        ),
    )

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.22/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    DecisionTreeRegressor = DecisionTreeRegressor.customize_schema(
        criterion={
            "description": "Function to measure the quality of a split.",
            "anyOf": [
                {"enum": ["mse", "friedman_mse", "poisson"]},
                {"enum": ["mae"], "forOptimizer": False},
            ],
            "default": "mse",
        },
        presort=None,
    )

lale.docstrings.set_docstrings(DecisionTreeRegressor)
