# Copyright 2023 IBM Corporation
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

import warnings

import pandas as pd
import sklearn.preprocessing

import lale.docstrings
import lale.lib.sklearn
import lale.operators
from lale.lib.imblearn._common_schemas import _hparam_n_jobs, _hparam_random_state

from ...helpers import with_fixed_estimator_name
from .orbis import Orbis
from .orbis import _hyperparams_schema as orbis_hyperparams_schema
from .util import (
    _categorical_fairness_properties,
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
    _validate_fairness_info,
)


def _orbis_schema(hparam):
    return orbis_hyperparams_schema["allOf"][0]["properties"][hparam]


class _BaggingOrbisClassifierImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        unfavorable_labels=None,
        redact=True,
        preparation=None,
        estimator=None,
        n_estimators=10,
        imbalance_repair_level=0.8,
        bias_repair_level=0.8,
        combine="keep_separate",
        sampling_strategy="mixed",
        replacement=False,
        n_jobs=None,
        random_state=None,
    ):
        assert unfavorable_labels is None, "not yet implemented"
        self.fairness_info = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "unfavorable_labels": unfavorable_labels,
        }
        _validate_fairness_info(**self.fairness_info, check_schema=False)
        self.redact = redact
        self.preparation = preparation
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.imbalance_repair_level = imbalance_repair_level
        self.bias_repair_level = bias_repair_level
        self.combine = combine
        self.sampling_strategy = sampling_strategy
        self.sampler_hparams = {
            "replacement": replacement,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "not yet implemented"
        # preemptively encode labels before BaggingClassifier does so
        self.lab_enc = sklearn.preprocessing.LabelEncoder().fit(y)
        fav_labels = list(
            self.lab_enc.transform(self.fairness_info["favorable_labels"])
        )
        if self.estimator is None:
            final_est = lale.lib.sklearn.DecisionTreeClassifier()
        else:
            final_est = self.estimator
        if self.preparation is None:
            prep_and_est = final_est
        else:
            prep_and_est = self.preparation >> final_est
        orbis = Orbis(
            favorable_labels=fav_labels,
            protected_attributes=self.fairness_info["protected_attributes"],
            estimator=prep_and_est,
            redact=self.redact,
            imbalance_repair_level=self.imbalance_repair_level,
            bias_repair_level=self.bias_repair_level,
            combine=self.combine,
            sampling_strategy=self.sampling_strategy,
            **self.sampler_hparams,
        )

        def _repair_dtypes(inner_X):  # for some reason BaggingClassifier spoils dtypes
            d = {
                col: pd.Series(inner_X[col], index=inner_X.index, dtype=typ, name=col)
                for col, typ in X.dtypes.items()
            }
            return pd.DataFrame(d)

        repair_dtypes = lale.lib.sklearn.FunctionTransformer(func=_repair_dtypes)
        trainable_ensemble = lale.lib.sklearn.BaggingClassifier(
            **with_fixed_estimator_name(
                estimator=repair_dtypes >> orbis,
                n_estimators=self.n_estimators,
                n_jobs=self.sampler_hparams["n_jobs"],
                random_state=self.sampler_hparams["random_state"],
            )
        )
        encoded_y = pd.Series(self.lab_enc.transform(y), index=y.index)
        self.trained_ensemble = trainable_ensemble.fit(X, encoded_y)
        return self

    def predict(self, X, **predict_params):
        with warnings.catch_warnings():
            # Bagging calls predict_proba on the trainable instead of the result of fit
            warnings.simplefilter("ignore", category=DeprecationWarning)
            encoded_y = self.trained_ensemble.predict(X, **predict_params)
        return self.lab_enc.inverse_transform(encoded_y)

    def predict_proba(self, X, **predict_params):
        with warnings.catch_warnings():
            # Bagging calls predict_proba on the trainable instead of the result of fit
            warnings.simplefilter("ignore", category=DeprecationWarning)
            result = self.trained_ensemble.predict_proba(X, **predict_params)
        return result


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [*_categorical_fairness_properties.keys()],
            "relevantToOptimizer": [
                "n_estimators",
                "imbalance_repair_level",
                "bias_repair_level",
            ],
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
                        {"description": "NoOp", "enum": [None]},
                    ],
                    "default": None,
                },
                "estimator": {
                    "description": "The nested classifier to fit on balanced subsets of the data.",
                    "anyOf": [
                        {"laleType": "operator"},
                        {"enum": [None], "description": "DecisionTreeClassifier"},
                    ],
                    "default": None,
                },
                "n_estimators": {
                    "description": "The number of base estimators in the ensemble.",
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 10,
                },
                "imbalance_repair_level": _orbis_schema("imbalance_repair_level"),
                "bias_repair_level": _orbis_schema("bias_repair_level"),
                "combine": _orbis_schema("combine"),
                "sampling_strategy": _orbis_schema("sampling_strategy"),
                "replacement": {
                    "description": "Whether under-sampling is with or without replacement.",
                    "type": "boolean",
                    "default": False,
                },
                "n_jobs": _hparam_n_jobs,
                "random_state": _hparam_random_state,
            },
        },
        {
            "description": "When sampling_strategy is minimum or maximum, both repair levels must be 1.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "sampling_strategy": {"not": {"enum": ["minimum", "maximum"]}}
                    },
                },
                {
                    "type": "object",
                    "properties": {
                        "imbalance_repair_level": {"enum": [1]},
                        "bias_repair_level": {"enum": [1]},
                    },
                },
            ],
        },
    ],
}


_combined_schemas = {
    "description": """Experimental BaggingOrbisClassifier in-estimator fairness mitigator.
Work in progress and subject to change; only supports pandas DataFrame so far.
Bagging ensemble classifier, where each inner classifier gets trained
on a subset of the data that has been balanced with `Orbis`_.
Unlike other mitigators in `lale.lib.aif360`, this mitigator does not
come from AIF360.

.. _`Orbis`: https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.orbis.html#lale.lib.aif360.orbis.Orbis
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.bagging_orbis_classifier.html#lale.lib.aif360.bagging_orbis_classifier.BaggingOrbisClassifier",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _categorical_supervised_input_fit_schema,
        "input_predict": _categorical_input_predict_schema,
        "output_predict": _categorical_output_predict_schema,
        "input_predict_proba": _categorical_input_predict_proba_schema,
        "output_predict_proba": _categorical_output_predict_proba_schema,
    },
}

BaggingOrbisClassifier = lale.operators.make_operator(
    _BaggingOrbisClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(BaggingOrbisClassifier)
