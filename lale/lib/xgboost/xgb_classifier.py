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

from sklearn.base import BaseEstimator

import lale.docstrings
import lale.operators

try:
    import xgboost

    xgboost_installed = True
except ImportError:
    xgboost_installed = False


class XGBClassifierImpl(BaseEstimator):
    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        verbosity=1,
        objective="binary:logistic",
        booster="gbtree",
        tree_method="auto",
        n_jobs=1,
        nthread=None,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        colsample_bynode=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        seed=None,
        missing=None,
        silent=None,
    ):
        assert xgboost_installed, """Your Python environment does not have xgboost installed. You can install it with
    pip install xgboost
or with
    pip install 'lale[full]'"""
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.seed = seed
        self.missing = missing
        self.silent = silent

    def fit(self, X, y, **fit_params):
        result = XGBClassifierImpl(
            self.max_depth,
            self.learning_rate,
            self.n_estimators,
            self.verbosity,
            self.objective,
            self.booster,
            self.tree_method,
            self.n_jobs,
            self.nthread,
            self.gamma,
            self.min_child_weight,
            self.max_delta_step,
            self.subsample,
            self.colsample_bytree,
            self.colsample_bylevel,
            self.colsample_bynode,
            self.reg_alpha,
            self.reg_lambda,
            self.scale_pos_weight,
            self.base_score,
            self.random_state,
            self.seed,
            self.missing,
            self.silent,
        )
        result._wrapped_model = xgboost.XGBClassifier(**self.get_params())
        result._wrapped_model.fit(X, y, **fit_params)
        return result

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)


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
                    "default": 10,
                    "minimum": 0,
                    "distribution": "uniform",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 20,
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
                    "default": 1000,
                    "minimumForOptimizer": 500,
                    "maximumForOptimizer": 1500,
                },
                "verbosity": {
                    "description": "The degree of verbosity.",
                    "type": "integer",
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
                    "type": "integer",
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
                    "type": "number",
                    "description": "Balancing of positive and negative weights.",
                    "default": 1,
                },
                "base_score": {
                    "type": "number",
                    "description": "The initial prediction score of all instances, global bias.",
                    "default": 0.5,
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random number seed.  (replaces seed)",
                    "default": 0,
                },
                "missing": {
                    "anyOf": [{"type": "number",}, {"enum": [None],}],
                    "default": None,
                    "description": "Value in the data which needs to be present as a missing value. If"
                    " If None, defaults to np.nan.",
                },
                "silent": {
                    "anyOf": [{"type": "boolean",}, {"enum": [None],}],
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
            "items": {"type": "array", "items": {"type": "number"},},
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
                {"type": "array", "items": {"type": "number"},},
                {"enum": [None]},
            ],
            "description": "Weight for each instance",
            "default": None,
        },
        "eval_set": {
            "anyOf": [{"type": "array",}, {"enum": [None],}],
            "default": None,
            "description": "A list of (X, y) pairs to use as a validation set for",
        },
        "sample_weight_eval_set": {
            "anyOf": [{"type": "array",}, {"enum": [None],}],
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
            "anyOf": [{"type": "integer",}, {"enum": [None],}],
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

lale.docstrings.set_docstrings(XGBClassifierImpl, _combined_schemas)

XGBClassifier = lale.operators.make_operator(XGBClassifierImpl, _combined_schemas)
