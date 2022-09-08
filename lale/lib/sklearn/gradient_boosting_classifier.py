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
    "description": "Gradient Boosting for classification.",
    "allOf": [
        {
            "type": "object",
            "required": ["init"],
            "relevantToOptimizer": [
                "loss",
                "n_estimators",
                "min_samples_split",
                "min_samples_leaf",
                "max_depth",
                "max_features",
            ],
            "additionalProperties": False,
            "properties": {
                "loss": {
                    "enum": ["deviance", "exponential"],
                    "default": "deviance",
                    "description": "The loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.",
                },
                "learning_rate": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.1,
                    "description": "learning rate shrinks the contribution of each tree by `learning_rate`.",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "The number of boosting stages to perform.",
                },
                "subsample": {
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximum": 1.0,
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 1.0,
                    "description": "The fraction of samples to be used for fitting the individual base learners.",
                },
                "criterion": {
                    "enum": ["friedman_mse", "mse", "mae"],
                    "default": "friedman_mse",
                    "description": "The function to measure the quality of a split.",
                },
                "min_samples_split": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 2,
                            "forOptimizer": False,
                            "distribution": "uniform",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "minimumForOptimizer": 0.01,
                            "maximumForOptimizer": 0.5,
                            "default": 0.05,
                        },
                    ],
                    "default": 2,
                    "description": "The minimum number of samples required to split an internal node:",
                },
                "min_samples_leaf": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "forOptimizer": False,
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 0.5,
                            "minimumForOptimizer": 0.01,
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
                "max_depth": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 5,
                    "default": 3,
                    "description": "Maximum depth of the individual regression estimators.",
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
                "init": {
                    "anyOf": [{"laleType": "operator"}, {"enum": ["zero", None]}],
                    "default": None,
                    "description": "An estimator object that is used to compute the initial predictions.",
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
                            "exclusiveMaximum": True,
                            "minimumForOptimizer": 0.01,
                            "default": 0.5,
                            "distribution": "uniform",
                        },
                        {"enum": ["auto", "sqrt", "log2", None]},
                    ],
                    "default": None,
                    "description": "The number of features to consider when looking for the best split.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Enable verbose output. If 1 then it prints progress and performance",
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
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.",
                },
                "presort": {
                    "anyOf": [{"type": "boolean"}, {"enum": ["auto"]}],
                    "default": "auto",
                    "description": "Whether to presort the data to speed up the finding of best splits in",
                },
                "validation_fraction": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                    "description": "The proportion of training data to set aside as validation set for early stopping.",
                },
                "n_iter_no_change": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 5,
                            "maximumForOptimizer": 10,
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "``n_iter_no_change`` is used to decide if early stopping will be used",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance for the early stopping. When the loss is not improving",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit the gradient boosting model.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples. Internally, it will be converted to",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "Target values (strings or integers in classification, real numbers",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Sample weights. If None, then samples are equally weighted. Splits",
        },
        "monitor": {
            "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
            "default": None,
            "description": "The monitor is called after each iteration with the current the current iteration, a reference to the estimator and the local variables of _fit_stages as keyword arguments callable(i, self, locals()).",
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
            "description": "The input samples. Internally, it will be converted to",
        },
    },
}
_output_predict_schema = {
    "description": "The predicted values.",
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
            "description": "The input samples. Internally, it will be converted to",
        },
    },
}
_output_predict_proba_schema = {
    "description": "The class probabilities of the input samples. The order of the",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
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
    "description": "Confidence scores for samples for each class in the model.",
    "anyOf": [
        {
            "description": "In the multi-way case, score per (sample, class) combination.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        {
            "description": "In the binary case, score for `self._classes[1]`.",
            "type": "array",
            "items": {"type": "number"},
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Gradient boosting classifier`_ random forest from scikit-learn.

.. _`Gradient boosting classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.gradient_boosting_classifier.html",
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
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

GradientBoostingClassifier: lale.operators.PlannedIndividualOp
GradientBoostingClassifier = lale.operators.make_operator(
    sklearn.ensemble.GradientBoostingClassifier, _combined_schemas
)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    # new: https://scikit-learn.org/0.22/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    from lale.schemas import AnyOf, Bool, Enum, Float

    GradientBoostingClassifier = GradientBoostingClassifier.customize_schema(
        presort=AnyOf(
            types=[Bool(), Enum(["deprecated", "auto"])],
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
        set_as_available=True,
    )

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.22/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    GradientBoostingClassifier = GradientBoostingClassifier.customize_schema(
        presort=None,
        criterion={
            "description": "Function to measure the quality of a split.",
            "anyOf": [
                {"enum": ["mse", "friedman_mse"]},
                {
                    "description": "Deprecated since version 0.24.",
                    "enum": ["mae"],
                    "forOptimizer": False,
                },
            ],
            "default": "friedman_mse",
        },
        set_as_available=True,
    )

if sklearn.__version__ >= "1.0":
    # old: https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    # new: https://scikit-learn.org/1.0/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    GradientBoostingClassifier = GradientBoostingClassifier.customize_schema(
        criterion={
            "description": """The function to measure the quality of a split.
Supported criteria are ‘friedman_mse’ for the mean squared error with improvement score by Friedman, ‘
squared_error’ for mean squared error, and ‘mae’ for the mean absolute error.
The default value of ‘friedman_mse’ is generally the best as it can provide a better approximation in some cases.""",
            "anyOf": [
                {"enum": ["squared_error", "friedman_mse"]},
                {
                    "description": "Deprecated since version 0.24 and 1.0.",
                    "enum": ["mae", "mse"],
                    "forOptimizer": False,
                },
            ],
            "default": "friedman_mse",
        },
        min_impurity_split=None,
        set_as_available=True,
    )

if sklearn.__version__ >= "1.1":
    # old: https://scikit-learn.org/1.0/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    # new: https://scikit-learn.org/1.1/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    GradientBoostingClassifier = GradientBoostingClassifier.customize_schema(
        loss={
            "description": "TThe loss function to be optimized. ‘log_loss’ refers to binomial and multinomial deviance, the same as used in logistic regression. It is a good choice for classification with probabilistic outputs. For loss ‘exponential’, gradient boosting recovers the AdaBoost algorithm.",
            "anyOf": [
                {"enum": ["log_loss", "exponential"]},
                {
                    "description": "Deprecated since version 1.1.",
                    "enum": ["deviance"],
                    "forOptimizer": False,
                },
            ],
            "default": "log_loss",
        },
        min_impurity_split=None,
        set_as_available=True,
    )

lale.docstrings.set_docstrings(GradientBoostingClassifier)
