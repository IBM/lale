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
import numpy as np

import lale.docstrings
import lale.lib.lale
import lale.operators
from lale.lib.imblearn._common_schemas import (
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc_under,
)

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
from ._mystic_util import calc_undersample_soln

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# This method assumes we have at most 9 classes and binary protected attributes
# (should revisit if these assumptions change)
def _pick_sizes(
    osizes: Dict[str, int], imbalance_repair_level: float, bias_repair_level: float, favorable_labels: Set[int]
) -> Dict[str, int]:
    # get class counts
    class_count_dict = {}
    for k, v in osizes.items():
        c = k[-1]
        if c not in class_count_dict:
            class_count_dict[c] = 0
        class_count_dict[c] += v
    
    # sorting by class count ensures that ci ratios will be <= 1
    sorted_by_count = sorted(class_count_dict.items(), key=lambda x: x[1])
    oci = []
    for i in range(len(sorted_by_count)-1):
        oci.append(sorted_by_count[i][1] / sorted_by_count[i+1][1])
    
    # if any class reordering has happened, update the group mapping and favorable_labels (for di calculations) accordingly
    class_mapping = {old:new for new, (old, _) in enumerate(sorted_by_count)}
    group_mapping = {k:k for k in osizes.keys()}
    for old, new in class_mapping:
        if int(old) in favorable_labels:
            favorable_labels.remove(int(old))
            favorable_labels.add(int(new))
        old_groups = list(filter(lambda x: x[-1] == old, group_mapping.keys()))
        for g in old_groups:
            group_mapping[g] = group_mapping[g][:-1] + new
    
    mapped_osizes = {k1: osizes[k2] for k1, k2 in group_mapping.items()}

    # calculate di ratios and invert if needed
    odi = []
    num_prot_attr = len(group_mapping.keys()[0])-1
    for pa in range(num_prot_attr):
        disadv_grp = list(filter(lambda x: x[pa] == "0", group_mapping.keys()))
        adv_grp = list(filter(lambda x: x[pa] == "1", group_mapping.keys()))
        disadv_grp_adv_cls = list(filter(lambda x: int(x[-1]) in favorable_labels, disadv_grp))
        disadv_grp_adv_cls_ct = sum(list(map(lambda x: mapped_osizes[x], disadv_grp_adv_cls)))
        disadv_grp_disadv_cls = list(filter(lambda x: int(x[-1]) not in favorable_labels, disadv_grp))
        disadv_grp_disadv_cls_ct = sum(list(map(lambda x: mapped_osizes[x] not in favorable_labels, disadv_grp_disadv_cls)))
        adv_grp_disadv_cls = list(filter(lambda x: int(x[-1]) in favorable_labels, adv_grp))
        adv_grp_disadv_cls_ct = list(filter(lambda x: mapped_osizes[x] not in favorable_labels, adv_grp_disadv_cls))
        adv_grp_adv_cls = list(filter(lambda x: int(x[-1]) not in favorable_labels, adv_grp))
        adv_grp_adv_cls_ct = list(filter(lambda x: mapped_osizes[x] in favorable_labels, adv_grp_adv_cls))
        calc_di = ((disadv_grp_adv_cls_ct) / (disadv_grp_adv_cls_ct + disadv_grp_disadv_cls_ct)) / ((adv_grp_adv_cls_ct) / (adv_grp_adv_cls_ct + adv_grp_disadv_cls_ct))
        if calc_di <= 1:
            odi.append(calc_di)
        else:
            odi.append(1/calc_di)
            for g in disadv_grp:
                group_mapping[g] = group_mapping[g][0:pa] + "1" + group_mapping[g][pa+1:]
            for g in adv_grp:
                group_mapping[g] = group_mapping[g][0:pa] + "0" + group_mapping[g][pa+1:]
    # recompute mapping based on any flipping of protected attribute values
    mapped_osizes = {k1: osizes[k2] for k1, k2 in group_mapping.items()}
    sorted_osizes = list(map(lambda x: x[1], sorted(mapped_osizes.items(), key=lambda x: x[0])))
    # construct variables for solver
    o_flat = np.array(sorted_osizes)
    oci_vec = np.array(oci).reshape(-1,1)
    nci_vec = oci_vec + imbalance_repair_level * (1 - oci_vec)
    odi_vec = np.array(odi).reshape(-1,1)
    ndi_vec = odi_vec + bias_repair_level * (1 - odi_vec)
    
    # pass into solver
    n_flat = calc_undersample_soln(o_flat, favorable_labels, nci_vec, ndi_vec)
    sorted_osize_keys = sorted(mapped_osizes.keys())
    mapped_nsize_tups = list(zip(sorted_osize_keys, n_flat))
    mapped_nsize_dict = {k:v for (k,v) in mapped_nsize_tups}
    nsizes = {g1:mapped_nsize_dict[g2] for g1,g2 in group_mapping.items()}

    return nsizes


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
        if self.hyperparams["sampling_strategy"] == "auto":
            inner_hyperparams = {
                **self.hyperparams,
                "sampling_strategy": _pick_sizes(
                    group_and_y.value_counts().sort_index().to_dict(),
                    self.imbalance_repair_level,
                    self.bias_repair_level,
                    set(self.favorable_labels)
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
