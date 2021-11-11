# Copyright 2020 IBM Corporation
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

from autoai_ts_libs.srom.estimators.time_series.models.MT2RForecaster import (  # type: ignore # noqa
    MT2RForecaster as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators


class _MT2RForecasterImpl:
    def __init__(
        self,
        target_columns,
        trend="Linear",
        residual="Linear",
        lookback_win="auto",
        prediction_win=12,
        n_jobs=-1,
    ):
        self._hyperparams = {
            "target_columns": target_columns,
            "trend": trend,
            "residual": residual,
            "lookback_win": lookback_win,
            "prediction_win": prediction_win,
            "n_jobs": n_jobs,
        }
        self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

    def fit(self, X, y):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X=None, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X=None):
        return self._wrapped_model.predict_proba(X)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "target_columns",
                "trend",
                "residual",
                "lookback_win",
                "prediction_win",
                "n_jobs",
            ],
            "relevantToOptimizer": ["trend", "residual"],
            "properties": {
                "target_columns": {
                    "description": """Column indices for columns to be forecasted.""",
                    "type": "array",
                    "items": {"type": "integer", "min": 0},  # Positive integers.
                    # "default": None, #TODO: This doesn't have a default value. is that ok?
                },
                "trend": {
                    "description": "Estimator to use for the trend in the data.",
                    "enum": ["Linear", "Mean", "Poly"],
                    "default": "Linear",
                },
                "residual": {
                    "description": "Estimator to use for the residuals",
                    "enum": ["Linear", "Difference", "GeneralizedMean"],
                    "default": "Linear",
                },
                "lookback_win": {
                    "description": "The number of time points in the window of data to use as predictors in the estimator.",
                    "anyOf": [{"enum": ["auto"]}, {"type": "integer"}],
                    "default": "auto",
                },
                "prediction_win": {
                    "description": "The number of time points to predict into the future. The estimator(s) will be trained to predict all of these time points.",
                    "type": "integer",
                    "default": 12,
                },
                "n_jobs": {
                    "description": "Number of processors to use when fitting a multivariate model.",
                    "anyOf": [
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of CPU cores.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": -1,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        },
        "y": {"laleType": "Any"},
    },
}

_input_predict_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["X"],
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"enum": [None]},
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        }
    },
}

_output_predict_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["X"],
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"enum": [None]},
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        }
    },
}

_output_predict_proba_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_ts_libs`_.

.. _`autoai_ts_libs`: https://pypi.org/project/autoai-ts-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.mt2r_forecaster.html",
    "import_from": "autoai_ts_libs.srom.estimators.time_series.models.MT2RForecaster",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "forecaster"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}

MT2RForecaster = lale.operators.make_operator(_MT2RForecasterImpl, _combined_schemas)

lale.docstrings.set_docstrings(MT2RForecaster)
