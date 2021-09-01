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

from autoai_ts_libs.srom.estimators.time_series.models.srom_estimators import (  # type: ignore # noqa
    DifferenceFlattenAutoEnsembler as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators


class _DifferenceFlattenAutoEnsemblerImpl:
    def __init__(
        self,
        feature_columns,
        target_columns,
        lookback_win,
        pred_win,
        time_column=-1,
        init_time_optimization=False,
        data_transformation_scheme=None,
        total_execution_time=2,
        execution_time_per_pipeline=1,
        dag_granularity="default",
        execution_platform="spark_node_random_search",
        n_leaders_for_ensemble=5,
        n_estimators_for_pred_interval=30,
        max_samples_for_pred_interval=1.0,
        multistep_prediction_strategy=None,
        multistep_prediction_win=1,
        ensemble_type="voting",
        multivariate_column_encoding=False,
        store_lookback_history=False,
        n_jobs=-1,
        estimator=None,
    ):
        self._hyperparams = {
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "lookback_win": lookback_win,
            "pred_win": pred_win,
            "time_column": time_column,
            "init_time_optimization": init_time_optimization,
            "data_transformation_scheme": data_transformation_scheme,
            "total_execution_time": total_execution_time,
            "execution_time_per_pipeline": execution_time_per_pipeline,
            "dag_granularity": dag_granularity,
            "execution_platform": execution_platform,
            "n_leaders_for_ensemble": n_leaders_for_ensemble,
            "n_estimators_for_pred_interval": n_estimators_for_pred_interval,
            "max_samples_for_pred_interval": max_samples_for_pred_interval,
            "multistep_prediction_strategy": multistep_prediction_strategy,
            "multistep_prediction_win": multistep_prediction_win,
            "ensemble_type": ensemble_type,
            "multivariate_column_encoding": multivariate_column_encoding,
            "store_lookback_history": store_lookback_history,
            "n_jobs": n_jobs,
            "estimator": estimator,
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
                "feature_columns",
                "target_columns",
                "lookback_win",
                "pred_win",
                "time_column",
                "init_time_optimization",
                "data_transformation_scheme",
                "total_execution_time",
                "execution_time_per_pipeline",
                "dag_granularity",
                "execution_platform",
                "n_leaders_for_ensemble",
                "n_estimators_for_pred_interval",
                "max_samples_for_pred_interval",
                "multistep_prediction_strategy",
                "multistep_prediction_win",
                "ensemble_type",
                "multivariate_column_encoding",
                "store_lookback_history",
                "n_jobs",
                "estimator",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "feature_columns": {
                    "description": """""",
                    "type": "array",
                    "items": {"type": "integer", "min": 0},  # Positive integers.
                    # "default": None, #TODO: This doesn't have a default value. is that ok?
                },
                "target_columns": {
                    "description": """Column indices for columns to be forecasted.""",
                    "type": "array",
                    "items": {"type": "integer", "min": 0},  # Positive integers.
                    # "default": None, #TODO: This doesn't have a default value. is that ok?
                },
                "lookback_win": {"description": "", "type": "integer"},
                "pred_win": {"description": "", "type": "integer"},
                "time_column": {
                    "description": "",
                    "type": "integer",
                    "default": -1,
                },
                "init_time_optimization": {
                    "description": "",
                    "type": "boolean",
                    "default": False,
                },
                "data_transformation_scheme": {
                    "description": "",
                    "enum": [
                        "auto",
                        "log",
                        "mean_division",
                        "mean_substraction",
                        "mean_division_log",
                        "mean_substraction_log",
                        "sqrt",
                        "reciprocal",
                        "anscombe",
                        "fisher",
                        None,
                    ],
                    "default": None,
                },
                "total_execution_time": {
                    "description": "",
                    "type": "integer",
                    "default": 2,
                },
                "execution_time_per_pipeline": {
                    "description": "",
                    "type": "integer",
                    "default": 1,
                },
                "dag_granularity": {
                    "description": "",
                    "enum": [
                        "default",
                        "tiny",
                        "flat",
                        "multioutput_tiny",
                        "multioutput_flat",
                        "regression_chain_flat",
                        "regression_chain_tiny",
                        "MIMO_flat",
                        "MINO_complete_flat",
                        "multi_xgboost",
                        "multi_lgbm",
                        "xgboost",
                        "lgbm",
                    ],
                    "default": "default",
                },
                "execution_platform": {
                    "description": "",
                    "enum": [
                        "spark_node_random_search"
                    ],  # TODO: are there any other values?
                    "default": "spark_node_random_search",
                },
                "n_leaders_for_ensemble": {
                    "description": "",
                    "type": "integer",
                    "default": 5,
                },
                "n_estimators_for_pred_interval": {
                    "description": "",
                    "type": "integer",
                    "default": 30,
                },
                "max_samples_for_pred_interval": {
                    "description": "",
                    "type": "number",
                    "default": 1.0,
                },
                "multistep_prediction_strategy": {
                    "description": "",
                    "enum": [None, "recursive", "multioutput"],
                    "default": None,
                },
                "multistep_prediction_win": {
                    "description": "",
                    "type": "integer",
                    "default": 1,
                },
                "ensemble_type": {
                    "description": "",
                    "enum": ["voting"],
                    "default": "voting",
                },
                "multivariate_column_encoding": {
                    "description": "",
                    "type": "boolean",
                    "default": False,
                },
                "store_lookback_history": {
                    "description": "",
                    "type": "boolean",
                    "default": False,
                },
                "n_jobs": {
                    "description": "",
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
                "estimator": {
                    "description": "estimator object",
                    "anyOf": [{"laleType": "Any"}, {"enum": [None]}],
                    "default": None,
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
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.difference_flatten_auto_ensembler.html",
    "import_from": "autoai_ts_libs.srom.estimators.time_series.models.srom_estimators",
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

DifferenceFlattenAutoEnsembler = lale.operators.make_operator(
    _DifferenceFlattenAutoEnsemblerImpl, _combined_schemas
)

lale.docstrings.set_docstrings(DifferenceFlattenAutoEnsembler)
