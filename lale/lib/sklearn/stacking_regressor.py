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
from sklearn.ensemble import StackingRegressor as SKLModel

import lale.docstrings
import lale.operators
from lale.lib.lale._common_schemas import schema_cv

from .stacking_utils import _concatenate_predictions_pandas


class _StackingRegressorImpl(SKLModel):
    def predict(self, X, **predict_params):
        return super().predict(X, **predict_params)

    def score(self, X, y, sample_weight=None):
        return super().score(X, y, sample_weight)

    def _concatenate_predictions(self, X, predictions):
        if not isinstance(X, pd.DataFrame):
            return super()._concatenate_predictions(X, predictions)
        return _concatenate_predictions_pandas(self, X, predictions)


_hyperparams_schema = {
    "description": "Stack of estimators with a final regressor.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimators",
                "final_estimator",
                "cv",
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
                    "description": "A regressor which will be used to combine the base estimators. The default classifier is a 'RidgeCV'",
                },
                "cv": schema_cv,
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
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values.",
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

_input_score_schema = {
    "description": "Return the coefficient of determination R^2 of the prediction.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape (n_samples, n_samples_fitted), where n_samples_fitted is the number of samples used in the fitting for the estimator.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "True values for 'X'.",
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
_output_score_schema = {
    "description": "R^2 of 'self.predict' wrt 'y'",
    "type": "number",
}


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Stacking regressor`_ from scikit-learn for stacking ensemble.

.. _`Stacking regressor`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.stacking_regressor.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_score_schema": _input_score_schema,
        "output_score_schema": _output_score_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

StackingRegressor: lale.operators.PlannedIndividualOp
StackingRegressor = lale.operators.make_operator(
    _StackingRegressorImpl, _combined_schemas
)

lale.docstrings.set_docstrings(StackingRegressor)
