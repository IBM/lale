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

from autoai_ts_libs.srom.estimators.time_series.models.T2RForecaster import (  # type: ignore # noqa
    T2RForecaster as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators


class _T2RForecasterImpl:
    def __init__(
        self,
        trend="Linear",
        residual="Linear",
        lookback_win="auto",
        prediction_win=12,
    ):
        self._hyperparams = {
            "trend": trend,
            "residual": residual,
            "lookback_win": lookback_win,
            "prediction_win": prediction_win,
        }
        self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

    def fit(self, X, y):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X=None, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["trend", "residual", "lookback_win", "prediction_win"],
            "relevantToOptimizer": ["trend", "residual"],
            "properties": {
                "trend": {
                    "description": "Estimator to use for the trend in the data.",
                    "enum": ["Linear", "Mean", "Poly"],
                    "default": "Linear",
                },
                "residual": {
                    "description": "Estimator to use for the residuals.",
                    "enum": ["Linear", "Difference", "GeneralizedMean"],
                    "default": "Linear",
                },
                "lookback_win": {
                    "description": "The number of time points in the window of data to use as predictors in the estimator.",
                    "enum": ["auto"],
                    "default": "auto",
                },
                "prediction_win": {
                    "description": "The number of time points to predict into the future. The estimator(s) will be trained to predict all of these time points.",
                    "type": "integer",
                    "default": 12,
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
    "required": ["X"],
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

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_ts_libs`_.

.. _`autoai_ts_libs`: https://pypi.org/project/autoai-ts-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.t2r_forecaster.html",
    "import_from": "autoai_ts_libs.srom.estimators.time_series.models.T2RForecaster",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "forecaster"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

T2RForecaster = lale.operators.make_operator(_T2RForecasterImpl, _combined_schemas)
lale.docstrings.set_docstrings(T2RForecaster)
