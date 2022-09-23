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
    "description": "A random forest classifier.",
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
                        {
                            "enum": [None],
                            "description": "Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
                        },
                    ],
                    "default": None,
                    "description": "The maximum depth of the tree.",
                },
                "min_samples_split": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 2,
                            "laleMaximum": "X/maxItems",  # number of rows
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 5,
                            "default": 2,
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
                            "minimumForOptimizer": 1,
                            "maximumForOptimizer": 5,
                            "default": 1,
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
                    "description": "The number of features to consider when looking for the best split.",
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
                    "description": "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.",
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
                    "description": "Threshold for early stopping in tree growth.",
                },
                "bootstrap": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.",
                },
                "oob_score": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use out-of-bag samples to estimate the generalization accuracy.",
                },
                "n_jobs": {
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
                    "description": "The number of jobs to run in parallel for both fit and predict.",
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
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Controls the verbosity when fitting and predicting.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.",
                },
                "class_weight": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": {"type": "number"}},
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                        },
                        {"enum": ["balanced", "balanced_subsample", None]},
                    ],
                    "description": "Weights associated with classes in the form ``{class_label: weight}``.",
                    "default": None,
                },
            },
        },
        {
            "description": "This classifier does not support sparse labels.",
            "type": "object",
            "laleNot": "y/isSparse",
        },
        {
            "description": "Out of bag estimation only available if bootstrap=True.",
            "anyOf": [
                {"type": "object", "properties": {"bootstrap": {"enum": [True]}}},
                {"type": "object", "properties": {"oob_score": {"enum": [False]}}},
            ],
        },
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
            "description": "The predicted classes.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "Samples are equally weighted."},
            ],
            "description": "Sample weights.",
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
        }
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
        }
    },
}

_output_predict_proba_schema = {
    "type": "array",
    "description": "The outer array is over samples aka rows.",
    "items": {
        "type": "array",
        "description": "The inner array has items corresponding to each class.",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Random forest classifier`_ from scikit-learn.

.. _`Random forest classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.random_forest_classifier.html",
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

RandomForestClassifier: lale.operators.PlannedIndividualOp
RandomForestClassifier = lale.operators.make_operator(
    sklearn.ensemble.RandomForestClassifier, _combined_schemas
)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    from lale.schemas import AnyOf, Float, Int, Null

    RandomForestClassifier = RandomForestClassifier.customize_schema(
        n_estimators=Int(
            desc="The number of trees in the forest.",
            minimum=1,
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
        set_as_available=True,
    )

if sklearn.__version__ >= "1.0":
    # old: https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # new: https://scikit-learn.org/1.0/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    from lale.schemas import AnyOf, Float, Int, Null

    RandomForestClassifier = RandomForestClassifier.customize_schema(
        min_impurity_split=None, set_as_available=True
    )

if sklearn.__version__ >= "1.1":
    # old: https://scikit-learn.org/1.0/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # new: https://scikit-learn.org/1.1/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    RandomForestClassifier = RandomForestClassifier.customize_schema(
        max_features={
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
            "default": "sqrt",
            "description": "The number of features to consider when looking for the best split.",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(RandomForestClassifier)
