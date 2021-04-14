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

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators

try:
    import xgboost  # type: ignore

    xgboost_installed = True
except ImportError:
    xgboost_installed = False
    if TYPE_CHECKING:
        import xgboost  # type: ignore


def _rename_one_feature(name):
    mapping = {"[": "&#91;", "]": "&#93;", "<": "&lt;"}
    for old, new in mapping.items():
        name = name.replace(old, new)
    return name


def _rename_all_features(X):
    if not isinstance(X, pd.DataFrame):
        return X
    mapped = [_rename_one_feature(f) for f in X.columns]
    if list(X.columns) == mapped:
        return X
    return pd.DataFrame(data=X, columns=mapped)


class _XGBClassifierImpl:
    _wrapped_model: xgboost.XGBClassifier

    def __init__(self, **hyperparams):
        assert xgboost_installed, """Your Python environment does not have xgboost installed. You can install it with
    pip install xgboost
or with
    pip install 'lale[full]'"""
        self._hyperparams = hyperparams

    def fit(self, X, y, **fit_params):
        result = _XGBClassifierImpl(**self._hyperparams)
        result._wrapped_model = xgboost.XGBClassifier(**self._hyperparams)
        renamed_X = _rename_all_features(X)
        result._wrapped_model.fit(renamed_X, y, **fit_params)
        return result

    def predict(self, X):
        renamed_X = _rename_all_features(X)
        result = self._wrapped_model.predict(renamed_X)
        return result

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


_hyperparams_schema = {
    "description": "Hyperparameter schema for a Lale wrapper for XGBoost.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "max_depth",
                "learning_rate",
                "n_estimators",
                "verbosity",
                "objective",
                "booster",
                "tree_method",
                "n_jobs",
                "gamma",
                "min_child_weight",
                "max_delta_step",
                "subsample",
                "colsample_bytree",
                "colsample_bylevel",
                "colsample_bynode",
                "reg_alpha",
                "reg_lambda",
                "scale_pos_weight",
                "base_score",
                "random_state",
                "missing",
            ],
            "relevantToOptimizer": [
                "gamma",
                "max_depth",
                "learning_rate",
                "n_estimators",
                "min_child_weight",
                "subsample",
                "reg_alpha",
                "reg_lambda",
            ],
            "properties": {
                "max_depth": {
                    "description": "Maximum tree depth for base learners.",
                    "type": "integer",
                    "default": 4,
                    "minimum": 0,
                    "distribution": "uniform",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 7,
                },
                "learning_rate": {
                    "description": "Boosting learning rate (xgb’s “eta”)",
                    "type": "number",
                    "default": 0.1,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 0.02,
                    "maximumForOptimizer": 1,
                },
                "n_estimators": {
                    "description": "Number of trees to fit.",
                    "type": "integer",
                    "default": 100,
                    "minimumForOptimizer": 50,
                    "maximumForOptimizer": 1000,
                },
                "verbosity": {
                    "description": "The degree of verbosity.",
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "minimum": 0,
                    "maximum": 3,
                },
                "objective": {
                    "description": "Specify the learning task and the corresponding "
                    "learning objective or a custom objective function to be used.",
                    "anyOf": [
                        {
                            "enum": [
                                "binary:logistic",
                                "binary:logitraw",
                                "binary:hinge",
                                "multi:softprob",
                                "multi:softmax",
                            ]
                        },
                        {"laleType": "callable"},
                    ],
                    "default": "binary:logistic",
                },
                "booster": {
                    "description": "Specify which booster to use.",
                    "enum": ["gbtree", "gblinear", "dart"],
                    "default": "gbtree",
                },
                "tree_method": {
                    "description": """Specify which tree method to use.
Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available.
Refer to https://xgboost.readthedocs.io/en/latest/parameter.html. """,
                    "enum": ["auto", "exact", "approx", "hist", "gpu_hist"],
                    "default": "auto",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "description": "Number of parallel threads used to run xgboost.  (replaces ``nthread``)",
                    "default": 1,
                },
                "nthread": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Number of parallel threads used to run xgboost.  Deprecated, please use n_jobs",
                },
                "gamma": {
                    "type": "number",
                    "description": "Minimum loss reduction required to make a further partition on a leaf node of the tree.",
                    "default": 0,
                    "minimum": 0,
                    "maximumForOptimizer": 1.0,
                },
                "min_child_weight": {
                    "type": "integer",
                    "description": "Minimum sum of instance weight(hessian) needed in a child.",
                    "default": 10,
                    "distribution": "uniform",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 20,
                },
                "max_delta_step": {
                    "type": "integer",
                    "description": "Maximum delta step we allow each tree's weight estimation to be.",
                    "default": 0,
                },
                "subsample": {
                    "type": "number",
                    "description": "Subsample ratio of the training instance.",
                    "default": 1,
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                },
                "colsample_bytree": {
                    "type": "number",
                    "description": "Subsample ratio of columns when constructing each tree.",
                    "default": 1,
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "maximum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.1,
                    "maximumForOptimizer": 1.0,
                },
                "colsample_bylevel": {
                    "type": "number",
                    "description": "Subsample ratio of columns for each split, in each level.",
                    "default": 1,
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "maximum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.1,
                    "maximumForOptimizer": 1.0,
                },
                "colsample_bynode": {
                    "type": "number",
                    "description": "Subsample ratio of columns for each split.",
                    "default": 1,
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "maximum": 1,
                },
                "reg_alpha": {
                    "type": "number",
                    "description": "L1 regularization term on weights",
                    "default": 0,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                },
                "reg_lambda": {
                    "type": "number",
                    "description": "L2 regularization term on weights",
                    "default": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.1,
                    "maximumForOptimizer": 1.0,
                },
                "scale_pos_weight": {
                    "anyOf": [
                        {
                            "type": "number",
                        },
                        {
                            "enum": [None],
                        },
                    ],
                    "description": "Balancing of positive and negative weights.",
                    "default": 1,
                },
                "base_score": {
                    "anyOf": [
                        {
                            "type": "number",
                        },
                        {
                            "enum": [None],
                        },
                    ],
                    "description": "The initial prediction score of all instances, global bias.",
                    "default": 0.5,
                },
                "random_state": {
                    "anyOf": [
                        {
                            "type": "integer",
                        },
                        {
                            "enum": [None],
                        },
                    ],
                    "description": "Random number seed.  (replaces seed)",
                    "default": 0,
                },
                "missing": {
                    "anyOf": [
                        {
                            "type": "number",
                        },
                        {
                            "enum": [None],
                        },
                    ],
                    "default": None,
                    "description": "Value in the data which needs to be present as a missing value. If"
                    " If None, defaults to np.nan.",
                },
                "silent": {
                    "anyOf": [
                        {
                            "type": "boolean",
                        },
                        {
                            "enum": [None],
                        },
                    ],
                    "default": None,
                    "description": "deprecated and replaced with verbosity, but adding to be backward compatible. ",
                },
                "seed": {
                    "default": None,
                    "description": "deprecated and replaced with random_state, but adding to be backward compatible. ",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit gradient boosting classifier",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Feature matrix",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "Labels",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "description": "Weight for each instance",
            "default": None,
        },
        "eval_set": {
            "anyOf": [
                {
                    "type": "array",
                },
                {
                    "enum": [None],
                },
            ],
            "default": None,
            "description": "A list of (X, y) pairs to use as a validation set for",
        },
        "sample_weight_eval_set": {
            "anyOf": [
                {
                    "type": "array",
                },
                {
                    "enum": [None],
                },
            ],
            "default": None,
            "description": "A list of the form [L_1, L_2, ..., L_n], where each L_i is a list of",
        },
        "eval_metric": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "string"},
                {"enum": [None]},
                {"type": "object"},
            ],
            "default": None,
            "description": "If a str, should be a built-in evaluation metric to use. See",
        },
        "early_stopping_rounds": {
            "anyOf": [
                {
                    "type": "integer",
                },
                {
                    "enum": [None],
                },
            ],
            "default": None,
            "description": "Activates early stopping. Validation error needs to decrease at",
        },
        "verbose": {
            "type": "boolean",
            "description": "If `verbose` and an evaluation set is used, writes the evaluation",
            "default": True,
        },
        "xgb_model": {
            "anyOf": [{"type": "string"}, {"enum": [None]}],
            "description": "file name of stored xgb model or 'Booster' instance Xgb model to be",
            "default": None,
        },
        "callbacks": {
            "anyOf": [{"type": "array", "items": {"type": "object"}}, {"enum": [None]}],
            "default": None,
            "description": "List of callback functions that are applied at each iteration. ",
        },
    },
}


_input_predict_schema = {
    "description": "Predict with `data`.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The dmatrix storing the input.",
        },
        "output_margin": {
            "type": "boolean",
            "default": False,
            "description": "Whether to output the raw untransformed margin value.",
        },
        "ntree_limit": {
            "anyOf": [{"type": "integer"}, {"enum": [None]}],
            "description": "Limit number of trees in the prediction; defaults to best_ntree_limit if defined",
        },
        "validate_features": {
            "type": "boolean",
            "default": True,
            "description": "When this is True, validate that the Booster's and data's feature_names are identical.",
        },
    },
}

_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}

_output_predict_proba_schema = {
    "description": "Probability of the sample for each class in the model.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`XGBClassifier`_ gradient boosted decision trees.

.. _`XGBClassifier`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.xgboost.xgb_classifier.html",
    "import_from": "xgboost",
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

XGBClassifier: lale.operators.PlannedIndividualOp
XGBClassifier = lale.operators.make_operator(_XGBClassifierImpl, _combined_schemas)

if xgboost_installed and xgboost.__version__ >= "0.90":
    # page 58 of https://readthedocs.org/projects/xgboost/downloads/pdf/release_0.90/
    import lale.schemas

    XGBClassifier = XGBClassifier.customize_schema(
        objective=lale.schemas.JSON(
            {
                "description": "Specify the learning task and the corresponding learning objective or a custom objective function to be used.",
                "anyOf": [
                    {
                        "enum": [
                            "binary:hinge",
                            "binary:logistic",
                            "binary:logitraw",
                            "multi:softmax",
                            "multi:softprob",
                        ]
                    },
                    {"laleType": "callable"},
                ],
                "default": "binary:logistic",
            }
        )
    )

if xgboost_installed and xgboost.__version__ >= "1.3":
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    XGBClassifier = XGBClassifier.customize_schema(
        monotone_constraints={
            "description": "Constraint of variable monotonicity.",
            "anyOf": [{"enum": [None]}, {"type": "string"}],
            "default": None,
        },
        interaction_constraints={
            "description": "Constraints for interaction representing permitted interactions. The constraints must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]], where each inner list is a group of indices of features that are allowed to interact with each other.",
            "anyOf": [{"enum": [None]}, {"type": "string"}],
            "default": None,
        },
        num_parallel_tree={
            "description": "Used for boosting random forest.",
            "anyOf": [{"enum": [None]}, {"type": "integer"}],
            "default": None,
        },
        validate_parameters={
            "description": "Give warnings for unknown parameter.",
            "anyOf": [{"enum": [None]}, {"type": "boolean"}, {"type": "integer"}],
            "default": None,
        },
        gpu_id={
            "description": "Device ordinal.",
            "anyOf": [
                {"type": "integer"},
                {"enum": [None]},
            ],
            "default": None,
        },
        max_depth={
            "description": "Maximum tree depth for base learners.",
            "anyOf": [
                {
                    "type": "integer",
                    "minimum": 0,
                    "distribution": "uniform",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 7,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        learning_rate={
            "description": """Boosting learning rate (xgb's "eta")""",
            "anyOf": [
                {
                    "type": "number",
                    "distribution": "loguniform",
                    "minimumForOptimizer": 0.02,
                    "maximumForOptimizer": 1,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        booster={
            "description": "Specify which booster to use.",
            "enum": ["gbtree", "gblinear", "dart", None],
            "default": None,
        },
        tree_method={
            "description": """Specify which tree method to use.
Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available.
Refer to https://xgboost.readthedocs.io/en/latest/parameter.html. """,
            "enum": ["auto", "exact", "approx", "hist", "gpu_hist", None],
            "default": None,
        },
        gamma={
            "description": "Minimum loss reduction required to make a further partition on a leaf node of the tree.",
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0,
                    "maximumForOptimizer": 1.0,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        min_child_weight={
            "description": "Minimum sum of instance weight(hessian) needed in a child.",
            "anyOf": [
                {
                    "type": "integer",
                    "distribution": "uniform",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 20,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        max_delta_step={
            "description": "Maximum delta step we allow each tree's weight estimation to be.",
            "anyOf": [{"enum": [None]}, {"type": "integer"}],
            "default": None,
        },
        subsample={
            "description": "Subsample ratio of the training instance.",
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        colsample_bytree={
            "description": "Subsample ratio of columns when constructing each tree.",
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "maximum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.1,
                    "maximumForOptimizer": 1.0,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        colsample_bylevel={
            "description": "Subsample ratio of columns for each split, in each level.",
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "maximum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.1,
                    "maximumForOptimizer": 1.0,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        colsample_bynode={
            "description": "Subsample ratio of columns for each split.",
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMinimum": True,
                    "maximum": 1,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        reg_alpha={
            "description": "L1 regularization term on weights",
            "anyOf": [
                {
                    "type": "number",
                    "distribution": "uniform",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        reg_lambda={
            "description": "L2 regularization term on weights",
            "anyOf": [
                {
                    "type": "number",
                    "distribution": "uniform",
                    "minimumForOptimizer": 0.1,
                    "maximumForOptimizer": 1,
                },
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        scale_pos_weight={
            "description": "Balancing of positive and negative weights.",
            "anyOf": [
                {"type": "number"},
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        base_score={
            "description": "The initial prediction score of all instances, global bias.",
            "anyOf": [
                {"type": "number"},
                {"enum": [None], "forOptimizer": False},
            ],
            "default": None,
        },
        importance_type={
            "description": "The feature importance type for the feature_importances_ property.",
            "enum": ["gain", "weight", "cover", "total_gain", "total_cover"],
            "default": "gain",
        },
        use_label_encoder={
            "description": """(Deprecated) Use the label encoder from scikit-learn to encode the labels.
            For new code, we recommend that you set this parameter to False.""",
            "type": "boolean",
            "default": True,
        },
        missing={
            "anyOf": [
                {
                    "type": "number",
                },
                {
                    "enum": [None, np.NaN],
                },
            ],
            "default": np.NaN,
            "description": "Value in the data which needs to be present as a missing value. If"
            " If None, defaults to np.nan.",
        },
        verbosity={
            "description": "The degree of verbosity.",
            "anyOf": [{"type": "integer"}, {"enum": [None]}],
            "default": None,
            "minimum": 0,
            "maximum": 3,
        },
    )

lale.docstrings.set_docstrings(XGBClassifier)
