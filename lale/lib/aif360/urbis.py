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

import logging
from typing import Dict, Set

import imblearn.under_sampling
import pandas as pd

import lale.docstrings
import lale.lib.lale
import lale.operators
from lale.lib.imblearn._common_schemas import (
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc_under,
)

from ._mystic_util import calc_undersample_soln, obtain_solver_info, parse_solver_soln
from .protected_attributes_encoder import ProtectedAttributesEncoder
from .redacting import Redacting
from .util import (
    _categorical_fairness_properties,
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
    _validate_fairness_info,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# This method assumes we have at most 9 classes and binary protected attributes
# (should revisit if these assumptions change)
def _pick_sizes(
    osizes: Dict[str, int],
    imbalance_repair_level: float,
    bias_repair_level: float,
    favorable_labels: Set[int],
) -> Dict[str, int]:
    group_mapping, o_flat, nci_vec, ndi_vec = obtain_solver_info(
        osizes, imbalance_repair_level, bias_repair_level, favorable_labels
    )

    # pass into solver
    n_flat = calc_undersample_soln(o_flat, favorable_labels, nci_vec, ndi_vec)

    return parse_solver_soln(n_flat, group_mapping)


class _UrbisImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        estimator,
        unfavorable_labels=None,
        redact=True,
        imbalance_repair_level=0.8,
        bias_repair_level=0.8,
        **hyperparams,
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, False
        )
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.estimator = estimator
        self.unfavorable_labels = unfavorable_labels
        self.redact = redact
        self.imbalance_repair_level = imbalance_repair_level
        self.bias_repair_level = bias_repair_level
        self.hyperparams = hyperparams

    def fit(self, X, y):
        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info, remainder="drop", combine="keep_separate"
        )
        encoded_X, encoded_y = prot_attr_enc.transform_X_y(X, y)
        encoded_Xy = pd.concat([encoded_X, encoded_y], axis=1)
        group_and_y = encoded_Xy.apply(
            lambda row: "".join([str(v) for v in row]), axis=1
        )
        assert X.shape[0] == group_and_y.shape[0]
        # TODO: Figure out a better workaround
        if isinstance(list(self.favorable_labels)[0], str):
            self.favorable_labels = set([1])
        if self.hyperparams["sampling_strategy"] == "auto":
            inner_hyperparams = {
                **self.hyperparams,
                "sampling_strategy": _pick_sizes(
                    group_and_y.value_counts().sort_index().to_dict(),
                    self.imbalance_repair_level,
                    self.bias_repair_level,
                    set(self.favorable_labels),
                ),
            }
        else:
            inner_hyperparams = self.hyperparams
        resampler = imblearn.under_sampling.RandomUnderSampler(**inner_hyperparams)
        Xy = pd.concat([X, y], axis=1)
        resampled_Xy, _ = resampler.fit_resample(Xy, group_and_y)
        resampled_X = resampled_Xy.iloc[:, :-1]
        resampled_y = resampled_Xy.iloc[:, -1]
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
            "relevantToOptimizer": ["imbalance_repair_level", "bias_repair_level"],
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
                "sampling_strategy": _hparam_sampling_strategy_anyof_neoc_under,
                "random_state": _hparam_random_state,
                "replacement": {
                    "description": "Whether the sample is with or without replacement.",
                    "type": "boolean",
                    "default": False,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """Urbis (Undersampling to Repair Bias and Imbalance Simultaneously) pre-estimator fairness mitigator.
Uses `RandomUnderSampler`_ to undersample not only members of the
majority class, but also members of privileged groups. Internally,
this works by replacing class labels by the cross product of classes
and groups, then downsampling new non-minority intersections.
Unlike other mitigators in `lale.lib.aif360`, this mitigator does not
come from AIF360.

.. _`RandomUnderSampler`: https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.fair_smotenc.html#lale.lib.aif360.urbis.Urbis",
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


Urbis = lale.operators.make_operator(_UrbisImpl, _combined_schemas)

lale.docstrings.set_docstrings(Urbis)
