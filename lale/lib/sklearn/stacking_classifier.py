# Copyright 2021 IBM Corporation
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

import pandas as pd
from sklearn.ensemble import StackingClassifier as SKLModel

import lale.docstrings
import lale.operators
from lale.lib.lale._common_schemas import schema_cv

from .stacking_utils import _concatenate_predictions_pandas


class _StackingClassifierImpl(SKLModel):
    def predict(self, X, **predict_params):
        return super().predict(X, **predict_params)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return super().score(X, y, sample_weight)

    def decision_function(self, X):
        return super().decision_function(X)

    def _concatenate_predictions(self, X, predictions):
        if not isinstance(X, pd.DataFrame):
            return super()._concatenate_predictions(X, predictions)
        return _concatenate_predictions_pandas(self, X, predictions)


_hyperparams_schema = {
    "description": "Stack of estimators with a final classifier.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimators",
                "final_estimator",
                "cv",
                "stack_method",
                "n_jobs",
                "passthrough",
            ],
            "relevantToOptimizer": [
                "estimators",
                "final_estimator",
                "cv",
                "passthrough",
            ],
            "additionalProperties": False,
            "properties": {
                "estimators": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "laleType": "tuple",
                        "items": [
                            {"type": "string"},
                            {"anyOf": [{"laleType": "operator"}, {"enum": [None]}]},
                        ],
                    },
                    "description": "Base estimators which will be stacked together. Each element of the list is defined as a tuple of string (i.e. name) and an estimator instance. An estimator can be set to ‘drop’ using set_params.",
                },
                "final_estimator": {
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                    "description": "A classifier which will be used to combine the base estimators. The default classifier is a 'LogisticRegression'",
                },
                "cv": schema_cv,
                "stack_method": {
                    "description": "Methods called for each base estimator. If ‘auto’, it will try to invoke, for each estimator, 'predict_proba', 'decision_function' or 'predict' in that order. Otherwise, one of 'predict_proba', 'decision_function' or 'predict'. If the method is not implemented by the estimator, it will raise an error.",
                    "default": "auto",
                    "enum": ["auto", "predict_proba", "decision_function", "predict"],
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of jobs to run in parallel for ``fit``.",
                },
                "passthrough": {
                    "type": "boolean",
                    "default": False,
                    "description": "When False, only the predictions of estimators will be used as training data for 'final_estimator'. When True, the 'final_estimator' is trained on the predictions as well as the original training data.",
                },
            },
        },
    ],
}
_input_fit_schema = {
    "description": "Fit the estimators.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "The target values (class labels).",
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
_input_transform_schema = {
    "description": "Fit to data, then transform it.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features",
        },
    },
}
_output_transform_schema = {
    "description": "Transformed array",
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "anyOf": [
                {"type": "number"},
                {"type": "array", "items": {"type": "number"}},
            ]
        },
    },
}

_input_predict_schema = {
    "description": "Predict target for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples.",
        },
    },
}
_output_predict_schema = {
    "description": "Predicted targets.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "description": "Predict class probabilities for X using 'final_estimator_.predict_proba'.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The input samples.",
        },
    },
}
_output_predict_proba_schema = {
    "description": "Class probabilities of the input samples.",
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
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_decision_function_schema = {
    "description": "The decision function computed by the final estimator.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Stacking classifier`_ from scikit-learn for stacking ensemble.

.. _`Stacking classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.stacking_classifier.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

StackingClassifier: lale.operators.PlannedIndividualOp
StackingClassifier = lale.operators.make_operator(
    _StackingClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(StackingClassifier)
