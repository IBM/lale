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

import imblearn.over_sampling
import numpy as np

import lale.docstrings
import lale.lib.lale
import lale.operators

from .redacting import Redacting
from .util import (
    _categorical_fairness_properties,
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
    _column_for_stratification,
    _validate_fairness_info,
)


class _FairSMOTEImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        estimator,
        unfavorable_labels=None,
        redact=True
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, False
        )
        if len(favorable_labels) != 1 or isinstance(favorable_labels[0], list):
            raise ValueError("favorable label must be unique, found {favorable_labels}")
        if unfavorable_labels is not None:
            if len(unfavorable_labels) != 1 or isinstance(unfavorable_labels[0], list):
                raise ValueError(
                    "unfavorable label must be unique, found {unfavorable_labels}"
                )
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.estimator = estimator
        self.unfavorable_labels = unfavorable_labels
        self.redact = redact

    def fit(self, X, y):
        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        groups_and_y = _column_for_stratification(X, y, **fairness_info)
        fav = self.favorable_labels[0]
        if self.unfavorable_labels is not None:
            unfav = self.unfavorable_labels[0]
        else:
            unfav = next(iter(set(y) - set(self.favorable_labels)))
        cats_mask = [not np.issubdtype(typ, np.number) for typ in X.dtypes]
        if all(cats_mask):  # all nominal -> use SMOTEN
            resampler = imblearn.over_sampling.SMOTEN(
                sampling_strategy="not majority",
                k_neighbors=5,
            )
        elif not any(cats_mask):  # all continuous -> use vanilla SMOTE
            resampler = imblearn.over_sampling.SMOTE(
                sampling_strategy="not majority",
                k_neighbors=5,
            )
        else:  # mix of nominal and continuous -> use SMOTENC
            resampler = imblearn.over_sampling.SMOTENC(
                categorical_features=cats_mask,
                sampling_strategy="not majority",
                k_neighbors=5,
            )
        resampled_X, resampled_groups_and_y = resampler.fit_resample(X, groups_and_y)
        resampled_y = resampled_groups_and_y.apply(
            lambda s: fav if s[-1] == "T" else unfav
        )
        if self.redact:
            redacting_trainable = Redacting(**fairness_info)
            self.redacting = redacting_trainable.fit(resampled_X)
        else:
            self.redacting = lale.lib.lale.NoOp
        redacted_X = self.redacting.transform(resampled_X)
        self.estimator = self.estimator.fit(redacted_X, resampled_y)
        return self

    def predict(self, X, **predict_params):
        redacted_X = self.redacting.transform(X)
        result = self.estimator.predict(redacted_X, **predict_params)
        return result

    def predict_proba(self, X, **predict_params):
        redacted_X = self.redacting.transform(X)
        result = self.estimator.predict_proba(redacted_X, **predict_params)
        return result


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "estimator",
                "redact",
            ],
            "relevantToOptimizer": [],
            "properties": {
                **_categorical_fairness_properties,
                "estimator": {
                    "description": "Nested classifier.",
                    "laleType": "operator",
                },
                "redact": {
                    "description": "Whether to redact protected attributes before data preparation (recommended) or not.",
                    "type": "boolean",
                    "default": True,
                },
            },
        }
    ],
}

_combined_schemas = {
    "description": """FairSMOTE pre-estimator fairness mitigator.
Uses `SMOTENC`_ (Synthetic Minority Over-sampling Technique for Nominal
and Continuous) to oversample not only members of the minority class,
but also members of unprivileged groups. Internally, this works by
replacing class labels by the cross product of classes and groups,
then upsampling all new non-majority "classes".
Unlike other mitigators in `lale.lib.aif360`, this mitigator does not
come from AIF360.

.. _`SMOTENC`: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.fair_smotenc.html#lale.lib.aif360.fair_smotenc.FairSMOTE",
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


FairSMOTE = lale.operators.make_operator(_FairSMOTEImpl, _combined_schemas)

lale.docstrings.set_docstrings(FairSMOTE)
