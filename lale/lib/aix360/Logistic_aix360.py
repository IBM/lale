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



import pandas as pd
from aix360.algorithms.rbm import LogisticRuleRegression, FeatureBinarizer

import lale.docstrings
import lale.operators

from .util import (
    dataset_to_pandas,
    _BaseInEstimatorImpl,
    _categorical_fairness_properties,
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)


class _logisticaix360Impl(_BaseInEstimatorImpl):
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        redact=True,
        preparation=None,
        lambda0=0.05,
        lambda1=0.01,
    ):
        mitigator = _LogisticAIXMitigatorImpl(lambda0=lambda0, lambda1=lambda1, encoding_func=self._prep_and_encode)
        super(_logisticaix360Impl, self).__init__(
            favorable_labels=favorable_labels,
            protected_attributes=protected_attributes,
            redact=redact,
            preparation=preparation,
            mitigator=mitigator,
        )

class _LogisticAIXMitigatorImpl:
    def __init__(self, lambda0, lambda1, encoding_func):
        self._mitigator = LogisticRuleRegression(lambda0=lambda0, lambda1=lambda1)
        self._encoding_func = encoding_func
    def fit(self, encoded_data):
        X, y = dataset_to_pandas(encoded_data)
        fb = FeatureBinarizer(negations=True)
        X_fb = fb.fit_transform(X)
        self._mitigator.fit(X_fb, y)
    def predict(self, encoded_data):
        X, y = dataset_to_pandas(encoded_data)
        #X, _ = dataset_to_pandas(encoded_data, return_only="X")
        fb = FeatureBinarizer(negations=True)
        X_fb = fb.fit_transform(X)
        y_pred = self._mitigator.predict(X_fb)
        y_pred = pd.Series(data=y_pred, index=X_fb.index,name=y.name)
        #y_pred = pd.Series(data=y_pred, index=X.index)
        encoded_result = self._encoding_func(X, y_pred)
        return encoded_result
    

_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema
_input_predict_proba_schema = _categorical_input_predict_proba_schema
_output_predict_proba_schema = _categorical_output_predict_proba_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "redact",
                "preparation",
                "lambda0",
                "lambda1",
            ],
            "relevantToOptimizer": ["lambda0", "lambda1"],
            "properties": {
                **_categorical_fairness_properties,
                "redact": {
                    "description": "Whether to redact protected attributes before data preparation (recommended) or not.",
                    "type": "boolean",
                    "default": True,
                },
                "preparation": {
                    "description": "Transformer, which may be an individual operator or a sub-pipeline.",
                    "anyOf": [
                        {"laleType": "operator"},
                        {"description": "lale.lib.lale.NoOp", "enum": [None]},
                    ],
                    "default": None,
                },
                "lambda0": {
                    "description": "Regularization - fixed cost of each rule.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "lambda1": {
                    "description": "Regularization - additional cost of each rule.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`Logistic_aix360`_ in-estimator fairness mitigator. Adds a discrimination-aware regularization term to the learning objective (`Kamishima et al. 2012`_).

.. _`Aix360 Logistic regression`: https://aix360.readthedocs.io/en/latest/dise.html#aix360.algorithms.rbm.logistic_regression.LogisticRuleRegression

""",
    "import_from": "aif360.sklearn.inprocessing",
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


logisticaix360 = lale.operators.make_operator(
    _logisticaix360Impl, _combined_schemas
)

lale.docstrings.set_docstrings(logisticaix360)