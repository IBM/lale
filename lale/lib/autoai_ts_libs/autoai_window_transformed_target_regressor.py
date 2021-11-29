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

from autoai_ts_libs.sklearn.mvp_windowed_transformed_target_estimators import (  # type: ignore # noqa
    AutoaiWindowTransformedTargetRegressor as model_to_be_wrapped,
)
from sklearn.pipeline import Pipeline, make_pipeline

import lale.docstrings
import lale.operators


class _AutoaiWindowTransformedTargetRegressorImpl:
    def __init__(
        self,
        feature_columns=None,
        target_columns=None,
        regressor=None,
        lookback_window=10,
        prediction_horizon=1,
        scaling_func=None,
        inverse_scaling_func=None,
        check_inverse=False,
        short_name="",
        one_shot=False,
        row_mean_center=False,
        estimator_prediction_type="forecast",
        time_column=-1,
        random_state=42,
    ):
        if regressor is None:
            nested_op = None
        elif isinstance(regressor, lale.operators.TrainableIndividualOp):
            nested_op = make_pipeline(regressor.impl)
        elif isinstance(regressor, lale.operators.BasePipeline):
            nested_op = regressor.export_to_sklearn_pipeline()
        elif isinstance(regressor, Pipeline):
            nested_op = regressor
        else:
            # TODO: What is the best way to handle this case?
            nested_op = None
        self._hyperparams = {
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "regressor": nested_op,
            "lookback_window": lookback_window,
            "prediction_horizon": prediction_horizon,
            "scaling_func": scaling_func,
            "inverse_scaling_func": inverse_scaling_func,
            "check_inverse": check_inverse,
            "short_name": short_name,
            "one_shot": one_shot,
            "row_mean_center": row_mean_center,
            "estimator_prediction_type": estimator_prediction_type,
            "time_column": time_column,
            "random_state": random_state,
        }
        self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

    def fit(self, X, y):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X=None, prediction_type=None, **predict_params):
        return self._wrapped_model.predict(X, prediction_type, **predict_params)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "feature_columns",
                "target_columns",
                "regressor",
                "lookback_window",
                "prediction_horizon",
                "scaling_func",
                "inverse_scaling_func",
                "check_inverse",
                "short_name",
                "one_shot",
                "row_mean_center",
                "estimator_prediction_type",
                "time_column",
                "random_state",
            ],
            "relevantToOptimizer": ["lookback_window"],
            "properties": {
                "feature_columns": {
                    "description": "",  # TODO: document and refine type
                    "laleType": "Any",
                    "default": None,
                },
                "target_columns": {
                    "description": "",  # TODO: document and refine type
                    "laleType": "Any",
                    "default": None,
                },
                "regressor": {
                    "description": """Regressor object.
This regressor will automatically be cloned each time prior to fitting.
If regressor is None, LinearRegression() is created and used.""",
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                },
                "lookback_window": {
                    "description": "The number of time points in the window of data to use as predictors in the estimator.",
                    "type": "integer",
                    "default": 10,
                },
                "prediction_horizon": {
                    "description": "The number of time points to predict into the future. The estimator(s) will be trained to predict all of these time points.",
                    "type": "integer",
                    "default": 1,
                },
                "scaling_func": {
                    "description": """(deprecated) Function to apply to y before passing to fit.
The function needs to return a 2-dimensional array.
If func is None, the function used will be the identity function.""",
                    "laleType": "Any",
                    "default": None,
                },
                "inverse_scaling_func": {
                    "description": """(deprecated) Function to apply to the prediction of the regressor.
The function needs to return a 2-dimensional array.
The inverse function is used to return predictions to the same space of the original training labels.""",
                    "laleType": "Any",
                    "default": None,
                },
                "check_inverse": {
                    "description": """Whether to check that transform followed by inverse_transform or func followed by inverse_func leads to the original targets.""",
                    "type": "boolean",
                    "default": False,
                },
                "short_name": {
                    "description": "Short name to be used for this estimator.",
                    "type": "string",
                    "default": "",
                },
                "one_shot": {
                    "description": "(deprecated)",
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": False,
                },
                "row_mean_center": {
                    "description": "Whether to apply the row mean center transformation to the data. If true, windows of the data according to lookback_window aree created and then those rows are normalized.",
                    "type": "boolean",
                    "default": False,
                },
                "estimator_prediction_type": {
                    "description": "Defines what predictions are returned by the predict functionality. Forecast: only generate predictions for the out of sample data, i.e., for the prediction window immediately following the input data. Rowwise: make predictions for each time point in the input data.",
                    "enum": [
                        "forecast",
                        "rowwise",
                    ],
                    "default": "forecast",
                },
                "time_column": {
                    "description": "",  # TODO: document and refine type
                    "laleType": "Any",
                    "default": -1,
                },
                "random_state": {
                    "description": "",  # TODO: document and refine type
                    "laleType": "Any",
                    "default": 42,
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
        },
        "prediction_type": {
            "enum": [
                "forecast",
                "rowwise",
            ]
        },
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
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.autoai_window_transformed_target_regressor.html",
    "import_from": "autoai_ts_libs.sklearn.mvp_windowed_transformed_target_estimators",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

AutoaiWindowTransformedTargetRegressor = lale.operators.make_operator(
    _AutoaiWindowTransformedTargetRegressorImpl, _combined_schemas
)

lale.docstrings.set_docstrings(AutoaiWindowTransformedTargetRegressor)
