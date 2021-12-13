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

from autoai_ts_libs.watfore.watfore_forecasters import (  # type: ignore # noqa
    WatForeForecaster as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators


class _WatForeForecasterImpl:
    def __init__(
        self,
        algorithm="hw",
        use_full_error_history=True,
        error_horizon_length=1,
        force_model=False,
        min_training_data=-1,
        p_min=0,
        p_max=-1,
        q_min=0,
        q_max=0,
        algorithm_type="additive",
        samples_per_season=2,
        initial_training_seasons=2,
        compute_seasonality=False,
        error_history_length=1,
        #        training_sample_size=0,
        box_cox_transform=False,
        ts_icol_loc=-1,
        prediction_horizon=1,
        log_transform=True,
        lookback_win=1,
        target_column_indices=-1,
        debug=False,
        **kwargs
    ):
        # code from Wes to fix a specific handling in autoai_ts:
        if "_" in algorithm:
            algorithm_parts = algorithm.split("_")
            algorithm = algorithm_parts[0]
            algorithm_type = algorithm_parts[1]
        self._hyperparams = {
            "algorithm": algorithm,
            "use_full_error_history": use_full_error_history,
            "error_horizon_length": error_horizon_length,
            "force_model": force_model,
            "min_training_data": min_training_data,
            "p_min": p_min,
            "p_max": p_max,
            "q_min": q_min,
            "q_max": q_max,
            "algorithm_type": algorithm_type,
            "samples_per_season": samples_per_season,
            "initial_training_seasons": initial_training_seasons,
            "compute_seasonality": compute_seasonality,
            "error_history_length": error_history_length,
            #            "training_sample_size":training_sample_size,
            "box_cox_transform": box_cox_transform,
            "ts_icol_loc": ts_icol_loc,
            "prediction_horizon": prediction_horizon,
            "log_transform": log_transform,
            "lookback_win": lookback_win,
            "target_column_indices": target_column_indices,
            "debug": debug,
        }
        self._hyperparams.update(**kwargs)
        self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

    def fit(self, X, y):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X=None, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X=None):
        return self._wrapped_model.predict_proba(X)

    def viz_label(self) -> str:
        return "WatForeForecaster_" + self._hyperparams["algorithm"]


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": True,
            "required": [
                "algorithm",
                "use_full_error_history",
                "error_horizon_length",
                "force_model",
                "min_training_data",
                "p_min",
                "p_max",
                "q_min",
                "q_max",
                "algorithm_type",
                "samples_per_season",
                "initial_training_seasons",
                "compute_seasonality",
                "error_history_length",
                #                "training_sample_size",
                "box_cox_transform",
                "ts_icol_loc",
                "prediction_horizon",
                "log_transform",
                "lookback_win",
                "target_column_indices",
                "debug",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "algorithm": {
                    "description": """Algorithm that is used to initialize the prediction model. Currently supported are 'hw' i.e. holtwinters,
'arima','bats', autoforecaster i.e., BATS model with Box-Conx transformation. Algorithm specific parameters
also need to be specified.
Additive and multiplicative variants of the Holt-Winters Seasonal forecasting method. This implementation of
the Holt-Winter algorithm variants assumes that the data it receives satisfy the above conditions.
Any pre-processing the data needs in order to satisfy the above assumptions should take place prior to model
updates and calls for prediction. This approach was followed in order to allow any type of pre-processing
(for example for filling missing values) on the data, independent of the H-W core calculations.
Implementation of BATS (Box-Cox transform, ARMA errors, Trend, and Seasonal components) Algorithms
Reference: Alysha M De Livera, Rob J Hyndman and Ralph D Snyder, "Forecasting time series with complex seasonal
patterns using exponential smoothing," Journal of the American Statistical Association (2011) 106(496), 1513-1527.
If algorithm = autoforecaster, This trains all models and to keep running statistics on their forecasting
errors as updates are requested. The error statistics are used to continually update the selection
of the best model, which is used to do the forecasts in the super class. The algorithm becomes initialized as
soon as the first algorithm becomes initialized so as to allow forecasts as soon as possible. It continues to
rate new algorithms as they become initialized and/or subsequent updates are applied.""",
                    "enum": [
                        "hw",
                        "arima",
                        "bats",
                        "autoforecaster",
                        "hw_additive",
                        "hw_multiplicative",
                    ],
                    "default": "hw",
                },
                "use_full_error_history": {
                    "description": """(ARIMA and HoltWinters (hw) ONLY) Trains arima model using full
error history from the data. If False, then only the last errorHorizonLength updates will be considered in the
values returned. The resulting instance:
1. does not force a model if suitable orders and/or coefficients can not be found. This can result in a model
which can not be initialized.
2.picks the required amount of training data automatically.
3.finds the AR order automatically.
4. finds the MA order automatically.
5.finds the difference order automatically.""",
                    "type": "boolean",
                    "default": True,
                },
                "error_horizon_length": {
                    "description": """(ARIMA ONLY) This parameter is used only when algroithm='arima' or watfore.Forecasters.arima, this is error horizon for
error in arima model.""",
                    "type": "integer",
                    "default": 1,
                },
                "force_model": {
                    "description": """(ARIMA ONLY) If True, then force the selection of a model based on the given orders, regardless of suitability. If False,
then the model may never become initialized if suitable coefficients for the training data can not be identified.""",
                    "type": "boolean",
                    "default": False,
                },
                "min_training_data": {
                    "description": """(ARIMA ONLY) The amount of training data to use to learn the coefficients. May be non-positive, in which case the minimum
amount of training data will be determined by the pMax and qMax values.""",
                    "type": "integer",
                    "default": -1,
                },
                "p_min": {
                    "description": """(ARIMA ONLY) Minimum AR order to be selected during training. Must be 0 or larger.""",
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                },
                "p_max": {
                    "description": """(ARIMA ONLY) Maximum AR order to be selected during training. If less than 0, then the maximum supported order will be used,
otherwise, must be at least as large as pMin.""",
                    "type": "integer",
                    "default": -1,
                },
                "q_min": {
                    "description": """(ARIMA ONLY) Minimum AR order to be selected during training. Must be 0 or larger.""",
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                },
                "q_max": {
                    "description": """(ARIMA ONLY) Maximum MA order to be selected during training. If less than 0, then the maximum supported order will be used,
otherwise, must be at least as large as qMin.""",
                    "type": "integer",
                    "default": 0,
                },
                "algorithm_type": {
                    "description": """(HoltWinters(hw) ONLY, i.e. when algorithm=watfore.Forecasters.hw or algorithm='hw')
`additive` provides implementation of the additive variant of the Holt-Winters Seasonal forecasting method.
The additive variant has the seasonal and trend/slope components enter the forecasting function in an
additive manner (see ref. 1), as in
See http://books.google.com/books?id=GSyzox8Lu9YC&source=gbs_navlinks_s for more information.
y(t+h) = L(t) + t*H(t) + S(t+h)
where
t = latest time for which the model has been updated
h = number of steps ahead for which a forecast is desired
L(t) = is the level estimate at time t
H(t) = is the slope at time t
S(t+h) = is the seasonal component at time t + h.
`multiplicative`, provides implementation of the multiplicative variant of the Holt-Winters Seasonal forecasting
method. The multiplicative variant has the seasonal and trend/slope components enter the forecasting function
in a multiplicative manner (see ref. 1, Brockwell, pp. 329).
y(t+h) = (L(t) + t* H(t)) * S(t+h)""",
                    "enum": ["additive", "multiplicative", None],
                    "default": "additive",
                },
                "samples_per_season": {
                    "description": """(hw ONLY)
Season length, or if compute_seasonality is True, then the maximum season to allow. If used as a maximum season,
then identification of the season length (and thus forecasting) can not happen until samples_per_season
* initial_training_seasons of data are provided.""",
                    "anyOf": [{"type": "integer"}, {"type": "number"}],
                    "default": 2,
                },
                "initial_training_seasons": {
                    "description": """(hw ONLY) Number of seasons to use in training. Must be greater than 1.""",
                    "type": "integer",
                    "default": 2,
                    "minimum": 2,
                },
                "compute_seasonality": {
                    "description": """(hw ONLY) If true then automatically compute seasonality.""",
                    "type": "boolean",
                    "default": False,
                },
                "error_history_length": {
                    "description": """ (hw ONLY)""",
                    "type": "integer",
                    "default": 1,
                },
                #                 "training_sample_size":{
                #                     "description":"""(BATS(bats) ONLY, i.e. when algorithm=watfore.Forecasters.bats or algorithm='bats')
                # training sample size, recommended #samples = 2*maximum_cycle_length""",
                #                     "type":"integer"
                #                 },
                "box_cox_transform": {
                    "description": """(BATS(bats) ONLY, i.e. when algorithm=watfore.Forecasters.bats or algorithm='bats')
Only estimate Box-Cox parameter if True; otherwise, ignore Box-Cox transform""",
                    "type": "boolean",
                    "default": False,
                },
                "ts_icol_loc": {
                    "description": """This parameter tells the forecasting model the absolute location of the timestamp column. For specifying
time stamp location put value in array e.g., [0] if 0th column is time stamp. The array is to support
multiple timestamps in future. If ts_icol_loc = -1 that means no timestamp is provided and all data is
time series. With ts_icol_loc=-1, the model will assume all the data is ordered and equally sampled.""",
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {"enum": [-1]},
                    ],
                    "default": -1,
                },
                "prediction_horizon": {
                    "description": "The number of time points to predict into the future.",
                    "type": "integer",
                    "default": 1,
                },
                "log_transform": {
                    "description": "Whether a log transform of the data should be applied before fitting the estimator.",
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": True,
                },
                "lookback_win": {
                    "description": "Not currently utilized.",
                    "type": "integer",
                    "default": 1,
                },
                "target_column_indices": {
                    "description": """This parameter tells the forecasting model the absolute location of the target column(s) that need to be
used for training model(s). While fiting the specified column(s) are used for training and are predicted
subsequently in the predict function. Default is -1 which will assume all columns except timestamp
are targets.""",
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {"enum": [-1]},
                    ],
                    "default": -1,
                },
                "debug": {"description": """""", "type": "boolean", "default": False},
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
    "import_from": "autoai_ts_libs.watfore.watfore_forecasters",
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

WatForeForecaster = lale.operators.make_operator(
    _WatForeForecasterImpl, _combined_schemas
)
lale.docstrings.set_docstrings(WatForeForecaster)
