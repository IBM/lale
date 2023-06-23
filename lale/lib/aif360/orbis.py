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
import warnings
from typing import Dict, Set

import imblearn.over_sampling
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import lale.docstrings
import lale.lib.lale
import lale.operators
from lale.lib.imblearn._common_schemas import (
    _hparam_n_jobs,
    _hparam_n_neighbors,
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc_over,
)

from ._mystic_util import calc_oversample_soln, obtain_solver_info, parse_solver_soln
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
    n_flat = calc_oversample_soln(o_flat, favorable_labels, nci_vec, ndi_vec)

    return parse_solver_soln(n_flat, group_mapping)


class _OrbisImpl:
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
        if len(favorable_labels) != 1 or isinstance(favorable_labels[0], list):
            raise ValueError(
                f"favorable label must be unique, found {favorable_labels}"
            )
        if unfavorable_labels is not None:
            if len(unfavorable_labels) != 1 or isinstance(unfavorable_labels[0], list):
                raise ValueError(
                    f"unfavorable label must be unique, found {unfavorable_labels}"
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
            **fairness_info, remainder="drop", combine="and"
        )
        encoded_X = prot_attr_enc.transform(X).reset_index(drop=True)
        lab_enc = LabelEncoder()
        encoded_y = pd.Series(lab_enc.fit_transform(y))
        label_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
        fav_set = set(label_mapping[x] for x in self.favorable_labels)
        encoded_Xy = pd.concat([encoded_X, encoded_y], axis=1, ignore_index=True)
        group_and_y = encoded_Xy.apply(
            lambda row: "".join([str(v) for v in row]), axis=1
        )
        assert X.shape[0] == group_and_y.shape[0]
        if self.hyperparams["sampling_strategy"] == "auto":
            inner_hyperparams = {
                **self.hyperparams,
                "sampling_strategy": _pick_sizes(
                    group_and_y.value_counts().sort_index().to_dict(),
                    self.imbalance_repair_level,
                    self.bias_repair_level,
                    fav_set,
                ),
            }
        else:
            inner_hyperparams = self.hyperparams
        if self.unfavorable_labels is not None:
            not_favorable_labels = self.unfavorable_labels
        else:
            not_favorable_labels = list(set(y) - set(self.favorable_labels))
        if len(not_favorable_labels) != 1:
            raise ValueError(
                f"unfavorable label must be unique, found {not_favorable_labels}"
            )
        cats_mask = [not np.issubdtype(typ, np.number) for typ in X.dtypes]
        if all(cats_mask):  # all nominal -> use SMOTEN
            resampler = imblearn.over_sampling.SMOTEN(**inner_hyperparams)
        elif not any(cats_mask):  # all continuous -> use vanilla SMOTE
            resampler = imblearn.over_sampling.SMOTE(**inner_hyperparams)
        else:  # mix of nominal and continuous -> use SMOTENC
            resampler = imblearn.over_sampling.SMOTENC(
                categorical_features=cats_mask, **inner_hyperparams
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            resampled_X, resampled_groups_and_y = resampler.fit_resample(X, group_and_y)
        resampled_y = resampled_groups_and_y.apply(
            lambda s: self.favorable_labels[0]
            if s[-1] == "1"
            else not_favorable_labels[0]
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
                "imbalance_repair_level": {
                    "description": "How much to repair for class imbalance (0 means original imbalance, 1 means perfect balance).",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                },
                "bias_repair_level": {
                    "description": "How much to repair for group bias (0 means original bias, 1 means perfect fairness).",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                },
                "sampling_strategy": _hparam_sampling_strategy_anyof_neoc_over,
                "random_state": _hparam_random_state,
                "k_neighbors": {
                    **_hparam_n_neighbors,
                    "description": "Number of nearest neighbours to use to construct synthetic samples.",
                    "default": 5,
                },
                "n_jobs": _hparam_n_jobs,
            },
        },
        {
            "description": "Can only tune repair levels for sampling_strategy='auto'.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"sampling_strategy": {"enum": ["auto"]}},
                },
                {
                    "type": "object",
                    "properties": {
                        "bias_repair_level": {"enum": [0.8]},
                        "imbalance_repair_level": {"enum": [0.8]},
                    },
                },
            ],
        },
    ],
}

_combined_schemas = {
    "description": """Orbis (Oversampling to Repair Bias and Imbalance Simultaneously) pre-estimator fairness mitigator.
Uses `SMOTENC`_ (Synthetic Minority Over-sampling Technique for Nominal
and Continuous) to oversample not only members of the minority class,
but also members of unprivileged groups. Internally, this works by
replacing class labels by the cross product of classes and groups,
then upsampling new non-majority intersections.
Unlike other mitigators in `lale.lib.aif360`, this mitigator does not
come from AIF360.

.. _`SMOTENC`: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.fair_smotenc.html#lale.lib.aif360.orbis.Orbis",
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


Orbis = lale.operators.make_operator(_OrbisImpl, _combined_schemas)

lale.docstrings.set_docstrings(Orbis)
