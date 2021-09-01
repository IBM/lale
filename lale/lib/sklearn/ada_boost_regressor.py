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

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor as SKLModel

import lale.docstrings
import lale.operators

from .fit_spec_proxy import _FitSpecProxy
from .function_transformer import FunctionTransformer


class _AdaBoostRegressorImpl:
    def __init__(
        self,
        base_estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        loss="linear",
        random_state=None,
    ):
        if base_estimator is None:
            estimator_impl = None
        else:
            estimator_impl = _FitSpecProxy(base_estimator)

        self._hyperparams = {
            "base_estimator": estimator_impl,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "loss": loss,
            "random_state": random_state,
        }
        self._wrapped_model = SKLModel(**self._hyperparams)
        self._hyperparams["base_estimator"] = base_estimator

    def get_params(self, deep=True):
        out = self._wrapped_model.get_params(deep=deep)
        # we want to return the lale operator, not the underlying impl
        out["base_estimator"] = self._hyperparams["base_estimator"]
        return out

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            feature_transformer = FunctionTransformer(
                func=lambda X_prime: pd.DataFrame(X_prime, columns=X.columns),
                inverse_func=None,
                check_inverse=False,
            )
            self._hyperparams["base_estimator"] = _FitSpecProxy(
                feature_transformer >> self._hyperparams["base_estimator"]
            )
            self._wrapped_model = SKLModel(**self._hyperparams)
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)

    def score(self, X, y, sample_weight=None):
        return self._wrapped_model.score(X, y, sample_weight)


_hyperparams_schema = {
    "description": "inherited docstring for AdaBoostRegressor    An AdaBoost regressor.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
                "n_estimators",
                "learning_rate",
                "loss",
                "random_state",
            ],
            "relevantToOptimizer": ["n_estimators", "learning_rate", "loss"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                    "description": "The base estimator from which the boosted ensemble is built.",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 50,
                    "maximumForOptimizer": 500,
                    "distribution": "uniform",
                    "default": 50,
                    "description": "The maximum number of estimators at which boosting is terminated.",
                },
                "learning_rate": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1.0,
                    "description": "Learning rate shrinks the contribution of each regressor by",
                },
                "loss": {
                    "enum": ["linear", "square", "exponential"],
                    "default": "linear",
                    "description": "The loss function to use when updating the weights after each",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator;",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Build a boosted regressor from the training set (X, y).",
    "required": ["X", "y"],
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The target values (real numbers).",
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
            "description": "Sample weights. If None, the sample weights are initialized to",
        },
    },
}
_input_predict_schema = {
    "description": "Predict regression value for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
    },
}
_output_predict_schema = {
    "description": "The predicted regression values.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`AdaBoost regressor`_ from scikit-learn for boosting ensemble.

.. _`AdaBoost regressor`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ada_boost_regressor.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


AdaBoostRegressor = lale.operators.make_operator(
    _AdaBoostRegressorImpl, _combined_schemas
)

lale.docstrings.set_docstrings(AdaBoostRegressor)
