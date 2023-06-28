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
import imblearn.under_sampling
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.preprocessing import LabelEncoder

import lale.docstrings
import lale.helpers
import lale.lib.lale
import lale.operators
from lale.lib.imblearn._common_schemas import _hparam_random_state

from ._mystic_util import (
    _calculate_ci_ratios,
    _calculate_di_ratios,
    calc_mixedsample_soln,
    calc_oversample_soln,
    calc_undersample_soln,
    obtain_solver_info,
    parse_solver_soln,
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# This method assumes we have at most 9 classes and binary protected attributes
# (should revisit if these assumptions change)
def _pick_sizes(
    osizes: Dict[str, int],
    imbalance_repair_level: float,
    bias_repair_level: float,
    favorable_labels: Set[int],
    sampling_strategy: str,
) -> Dict[str, int]:
    if sampling_strategy in ["minimum", "maximum"]:
        assert imbalance_repair_level == 1, imbalance_repair_level
        assert bias_repair_level == 1, bias_repair_level
        if sampling_strategy == "minimum":
            one_size_fits_all = min(osizes.values())
        else:
            one_size_fits_all = max(osizes.values())
        nsizes = {k: one_size_fits_all for k in osizes.keys()}
    else:
        group_mapping, o_flat, nci_vec, ndi_vec = obtain_solver_info(
            osizes, imbalance_repair_level, bias_repair_level, favorable_labels
        )
        if sampling_strategy == "under":
            n_flat = calc_undersample_soln(o_flat, favorable_labels, nci_vec, ndi_vec)
        elif sampling_strategy == "over":
            n_flat = calc_oversample_soln(o_flat, favorable_labels, nci_vec, ndi_vec)
        elif sampling_strategy == "mixed":
            n_flat = calc_mixedsample_soln(o_flat, favorable_labels, nci_vec, ndi_vec)
        else:
            assert False, f"unexpected sampling_strategy {sampling_strategy}"
        nsizes = parse_solver_soln(n_flat, group_mapping)
        obtained_ci = np.array(_calculate_ci_ratios(nsizes)).reshape(-1, 1)
        obtained_di = np.array(
            _calculate_di_ratios(nsizes, favorable_labels, symmetric=True)
        ).reshape(-1, 1)
        assert_allclose(obtained_ci, nci_vec, rtol=0.05)
        assert_allclose(obtained_di, ndi_vec, rtol=0.05)
    return nsizes


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
        combine="keep_separate",
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
        self.combine = combine
        self.hyperparams = hyperparams

    def fit(self, X, y):
        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info, remainder="drop", combine=self.combine
        )
        encoded_X = prot_attr_enc.transform(X)
        lab_enc = LabelEncoder()
        encoded_y = pd.Series(lab_enc.fit_transform(y), index=y.index)
        label_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
        fav_set = set(label_mapping[x] for x in self.favorable_labels)
        encoded_Xy = pd.concat([encoded_X, encoded_y], axis=1, ignore_index=True)
        diaeresis_y = encoded_Xy.apply(
            lambda row: "".join([str(v) for v in row]), axis=1
        ).rename("diaeresis_y")
        assert X.shape[0] == diaeresis_y.shape[0]
        osizes = diaeresis_y.value_counts().sort_index().to_dict()
        nsizes = _pick_sizes(
            osizes,
            self.imbalance_repair_level,
            self.bias_repair_level,
            fav_set,
            self.hyperparams["sampling_strategy"],
        )
        Xyy = pd.concat([X, y, diaeresis_y], axis=1)
        # under-sample
        under_sizes = {k: min(ns, osizes[k]) for k, ns in nsizes.items()}
        under_hparams = {**self.hyperparams, "sampling_strategy": under_sizes}
        under_op = imblearn.under_sampling.RandomUnderSampler(**under_hparams)
        under_Xyy_all, _ = under_op.fit_resample(Xyy, diaeresis_y)
        shrunk_labels = [k for k, ns in nsizes.items() if ns < osizes[k]]
        under_Xyy = under_Xyy_all[under_Xyy_all.iloc[:, -1].isin(shrunk_labels)]
        # over-sample
        over_sizes = {k: max(ns, osizes[k]) for k, ns in nsizes.items()}
        over_hparams = {
            **lale.helpers.dict_without(self.hyperparams, "replacement"),
            "sampling_strategy": over_sizes,
        }
        cats_mask = [not np.issubdtype(typ, np.number) for typ in Xyy.dtypes]
        if all(cats_mask):  # all nominal -> use SMOTEN
            over_op = imblearn.over_sampling.SMOTEN(**over_hparams)
        elif not any(cats_mask):  # all continuous -> use vanilla SMOTE
            over_op = imblearn.over_sampling.SMOTE(**over_hparams)
        else:  # mix of nominal and continuous -> use SMOTENC
            over_op = imblearn.over_sampling.SMOTENC(
                categorical_features=cats_mask, **over_hparams
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            over_Xyy_all, _ = over_op.fit_resample(Xyy, diaeresis_y)
        not_shrunk_labels = [k for k in nsizes if k not in shrunk_labels]
        over_Xyy = over_Xyy_all[over_Xyy_all.iloc[:, -1].isin(not_shrunk_labels)]
        # shuffle and redact
        resampled_Xyy = pd.concat([under_Xyy, over_Xyy], axis=0).sample(frac=1)
        resampled_X = resampled_Xyy.iloc[:, :-2]
        resampled_y = resampled_Xyy.iloc[:, -2]
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
                "combine": {
                    "description": "How to handle the case when there is more than one protected attribute.",
                    "enum": ["keep_separate", "and", "or", "error"],
                    "default": "keep_separate",
                },
                "sampling_strategy": {
                    "enum": ["under", "over", "mixed", "minimum", "maximum"],
                    "description": """How to change the intersection sizes.
Possible choices are:

- ``'under'``: under-sample large intersections to desired repair levels;
- ``'over'``: over-sample small intersection to desired repair levels;
- ``'mixed'``: mix under- with over-sampling while keeping sizes similar to original;
- ``'minimum'``: under-sample everything to the size of the smallest intersection;
- ``'maximum'``: over-sample everything to the size of the largest intersection.""",
                    "default": "mixed",
                },
                "random_state": _hparam_random_state,
                "replacement": {
                    "description": "Whether under-sampling is with or without replacement.",
                    "type": "boolean",
                    "default": False,
                },
            },
        },
        {
            "description": "For sampling_strategy is minimum or maximum, both repair levels must be 1.",
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
    "description": """Orbis (Undersampling to Repair Bias and Imbalance Simultaneously) pre-estimator fairness mitigator.
Uses `SMOTE`_ and `RandomUnderSampler`_ to resample not only for
repairing class imbalance, but also group bias.
Internally, this works by replacing class labels by the cross product
of classes and groups, then changing the sizes of the new
intersections to achieve the desired repair levels.
Unlike other mitigators in `lale.lib.aif360`, this mitigator does not
come from AIF360.

.. _`RandomUnderSampler`: https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
.. _`SMOTE`: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
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
