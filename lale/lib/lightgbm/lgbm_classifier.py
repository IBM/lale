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

import lale.docstrings
import lale.operators

try:
    import lightgbm.sklearn

    lightgbm_installed = True
except ImportError:
    lightgbm_installed = False


class _LGBMClassifierImpl:
    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=-1,
        silent=True,
        importance_type="split",
    ):
        assert lightgbm_installed, """Your Python environment does not have lightgbm installed. You can install it with
    pip install lightgbm
or with
    pip install 'lale[full]'"""
        self._hyperparams = {
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample_for_bin": subsample_for_bin,
            "objective": objective,
            "class_weight": class_weight,
            "min_split_gain": min_split_gain,
            "min_child_weight": min_child_weight,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "silent": silent,
            "importance_type": importance_type,
        }
        self._wrapped_model = lightgbm.sklearn.LGBMClassifier(**self._hyperparams)

    def fit(self, X, y=None, **fit_params):
        if X.shape[0] * self._wrapped_model.subsample < 1.0:
            self._wrapped_model.subsample = 1.001 / X.shape[0]
        try:
            self._wrapped_model.fit(X, y, **fit_params)
        except Exception as e:
            raise RuntimeError(str(self._hyperparams)) from e
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


_hyperparams_schema = {
    "description": "LightGBM classifier. (https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "boosting_type",
                "max_depth",
                "learning_rate",
                "n_estimators",
                "min_child_samples",
                "subsample",
                "subsample_freq",
            ],
            "relevantToOptimizer": [
                "boosting_type",
                "num_leaves",
                "learning_rate",
                "n_estimators",
                "min_child_weight",
                "min_child_samples",
                "subsample",
                "subsample_freq",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            ],
            "additionalProperties": False,
            "properties": {
                "boosting_type": {
                    "anyOf": [
                        {"enum": ["gbdt", "dart"]},
                        {"enum": ["goss", "rf"], "forOptimizer": False},
                    ],
                    "default": "gbdt",
                    "description": "‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.",
                },
                "num_leaves": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {"enum": [2, 4, 8, 32, 64, 128, 16]},
                    ],
                    "default": 31,
                    "description": "Maximum tree leaves for base learners",
                },
                "max_depth": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 5,
                        }
                    ],
                    "default": -1,
                    "description": "Maximum tree depth for base learners, <=0 means no limit",
                },
                "learning_rate": {
                    "type": "number",
                    "minimumForOptimizer": 0.02,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.1,
                    "description": "Boosting learning rate.",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 50,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 200,
                    "description": "Number of boosted trees to fit.",
                },
                "subsample_for_bin": {
                    "type": "integer",
                    "default": 200000,
                    "description": "Number of samples for constructing bins.",
                },
                "objective": {
                    "anyOf": [
                        {"type": "object"},
                        {"enum": ["binary", "multiclass", None]},
                    ],
                    "default": None,
                    "description": "Specify the learning task and the corresponding learning objective or a custom objective function to be used",
                },
                "class_weight": {
                    "anyOf": [{"type": "object"}, {"enum": ["balanced", None]}],
                    "default": None,
                    "description": "Weights associated with classes",
                },
                "min_split_gain": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Minimum loss reduction required to make a further partition on a leaf node of the tree.",
                },
                "min_child_weight": {
                    "type": "number",
                    "minimumForOptimizer": 0.0001,
                    "maximumForOptimizer": 0.01,
                    "default": 1e-3,
                    "description": "Minimum sum of instance weight (hessian) needed in a child (leaf).",
                },
                "min_child_samples": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 30,
                    "distribution": "uniform",
                    "default": 20,
                    "description": "Minimum number of data needed in a child (leaf).",
                },
                "subsample": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "exclusiveMinimum": True,
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 1.0,
                    "description": "Subsample ratio of the training instance.",
                },
                "subsample_freq": {
                    "type": "integer",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 5,
                    "distribution": "uniform",
                    "default": 0,
                    "description": "Frequence of subsample, <=0 means no enable.",
                },
                "colsample_bytree": {
                    "type": "number",
                    "default": 1.0,
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "description": "Subsample ratio of columns when constructing each tree.",
                },
                "reg_alpha": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "default": 0.0,
                    "description": "L1 regularization term on weights.",
                },
                "reg_lambda": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "default": 0.0,
                    "description": "L2 regularization term on weights.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Random number seed. If None, default seeds in C++ code will be used.",
                },
                "n_jobs": {
                    "type": "integer",
                    "default": -1,
                    "description": "Number of parallel threads.",
                },
                "silent": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to print messages while running boosting.",
                },
                "importance_type": {
                    "enum": ["split", "gain"],
                    "default": "split",
                    "description": "The type of feature importance to be filled into feature_importances_.",
                },
            },
        },
        {
            "description": "boosting_type `rf` needs bagging (which means subsample_freq > 0 and subsample < 1.0)",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"boosting_type": {"not": {"enum": ["rf"]}}},
                },
                {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"subsample_freq": {"not": {"enum": [0]}}},
                        },
                        {
                            "type": "object",
                            "properties": {"subsample": {"not": {"enum": [1.0]}}},
                        },
                    ]
                },
            ],
        },
        {
            "description": "boosting_type `goss` can not use bagging (which means subsample_freq = 0 and subsample = 1.0)",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"boosting_type": {"not": {"enum": ["goss"]}}},
                },
                {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"subsample_freq": {"enum": [0]}},
                        },
                        {
                            "type": "object",
                            "properties": {"subsample": {"enum": [1.0]}},
                        },
                    ]
                },
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Build a lightgbm model from the training set (X, y).",
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
            "default": None,
            "description": "Weights of training data.",
        },
        "init_score": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Init score of training data.",
        },
        "group": {"default": None, "description": "Group data of training data."},
        "eval_set": {
            "default": None,
            "description": "A list of (X, y) tuple pairs to use as validation sets.",
        },
        "eval_names": {"default": None, "description": "Names of eval_set."},
        "eval_sample_weight": {"default": None, "description": "Weights of eval data."},
        "eval_class_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Class weights of eval data.",
        },
        "eval_init_score": {"default": None, "description": "Init score of eval data."},
        "eval_group": {"default": None, "description": "Group data of eval data."},
        "eval_metric": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"enum": ["logloss", None]},
                {"laleType": "callable"},
            ],
            "default": None,
            "description": "string, list of strings, callable or None, optional (default=None).",
        },
        "early_stopping_rounds": {
            "anyOf": [{"type": "integer"}, {"enum": [None]}],
            "default": None,
            "description": "Activates early stopping. The model will train until the validation score stops improving.",
        },
        "verbose": {
            "anyOf": [{"type": "boolean"}, {"type": "integer"}],
            "default": True,
            "description": "Requires at least one evaluation data.",
        },
        "feature_name": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"enum": ["auto"]},
            ],
            "default": "auto",
            "description": "Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.",
        },
        "categorical_feature": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                },
                {"enum": ["auto"]},
            ],
            "default": "auto",
            "description": "Categorical features. If list of int, interpreted as indices. If list of strings, interpreted as feature names.",
        },
        "callbacks": {
            "anyOf": [{"type": "array", "items": {"type": "object"}}, {"enum": [None]}],
            "default": None,
            "description": "List of callback functions that are applied at each iteration. ",
        },
    },
}
_input_predict_schema = {
    "description": "Return the predicted value for each sample.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": " Input features matrix.",
        },
        "raw_score": {
            "type": "boolean",
            "default": False,
            "description": "Whether to predict raw scores.",
        },
        "num_iteration": {
            "anyOf": [{"type": "integer"}, {"enum": [None]}],
            "default": None,
            "description": "Limit number of iterations in the prediction.",
        },
        "pred_leaf": {
            "type": "boolean",
            "default": False,
            "description": "Whether to predict leaf index.",
        },
        "pred_contrib": {
            "type": "boolean",
            "default": False,
            "description": "Whether to predict feature contributions.",
        },
    },
}

_output_predict_schema = {
    "description": "Return the predicted value for each sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Return the predicted probability for each class for each sample.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": " Input features matrix.",
        },
        "raw_score": {
            "type": "boolean",
            "default": False,
            "description": "Whether to predict raw scores.",
        },
        "num_iteration": {
            "anyOf": [{"type": "integer"}, {"enum": [None]}],
            "default": None,
            "description": "Limit number of iterations in the prediction.",
        },
        "pred_leaf": {
            "type": "boolean",
            "default": False,
            "description": "Whether to predict leaf index.",
        },
        "pred_contrib": {
            "type": "boolean",
            "default": False,
            "description": "Whether to predict feature contributions.",
        },
    },
}
_output_predict_proba_schema = {
    "description": "Return the predicted probability for each class for each sample.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lightgbm.lgbm_classifier.html",
    "import_from": "lightgbm.sklearn",
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


LGBMClassifier = lale.operators.make_operator(_LGBMClassifierImpl, _combined_schemas)

lale.docstrings.set_docstrings(LGBMClassifier)
