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
from typing import Dict, Set

import imblearn.over_sampling
import imblearn.under_sampling
import numpy as np
import pandas as pd
import sklearn.preprocessing
from numpy.testing import assert_allclose

import lale.docstrings
import lale.lib.lale
import lale.operators
from lale.lib.imblearn._common_schemas import (
    _hparam_n_jobs,
    _hparam_n_neighbors,
    _hparam_random_state,
)

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


def _make_diaeresis(X, y, fairness_info, combine):
    prot_attr_enc = ProtectedAttributesEncoder(
        **fairness_info, remainder="drop", combine=combine
    )
    encoded_X = prot_attr_enc.transform(X)
    lab_enc = sklearn.preprocessing.LabelEncoder().fit(y)
    encoded_y = pd.Series(lab_enc.transform(y), index=y.index)
    encoded_Xy = pd.concat([encoded_X, encoded_y], axis=1, ignore_index=True)
    diaeresis_y = encoded_Xy.apply(
        lambda row: "".join([str(v) for v in row]), axis=1
    ).rename("diaeresis_y")
    assert X.shape[0] == diaeresis_y.shape[0]
    fav_set = set(lab_enc.transform(fairness_info["favorable_labels"]))
    return diaeresis_y, fav_set


# This method assumes we have at most 9 classes and binary protected attributes
# (should revisit if these assumptions change)
def _orbis_pick_sizes(
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


def _orbis_resample(X, y, diaeresis_y, osizes, nsizes, sampler_hparams):
    # concat y so we can get it back out without needing an inverse to diaeresis
    # concat diaeresis_y so we can filter on it after resampling
    Xyy = pd.concat([X, y, diaeresis_y], axis=1)
    # under-sample entire data, then keep only shrunk labels
    under_sizes = {k: min(ns, osizes[k]) for k, ns in nsizes.items()}
    under_hparams = {
        **{
            h: v
            for h, v in sampler_hparams.items()
            if h not in ["k_neighbors", "n_jobs"]
        },
        "sampling_strategy": under_sizes,
    }
    under_op = imblearn.under_sampling.RandomUnderSampler(**under_hparams)
    under_Xyy_all, _ = under_op.fit_resample(Xyy, diaeresis_y)
    shrunk_labels = [k for k, ns in nsizes.items() if ns < osizes[k]]
    under_Xyy = under_Xyy_all[under_Xyy_all.iloc[:, -1].isin(shrunk_labels)]
    # over-sample entire data, then keep only not-shrunk labels
    over_sizes = {k: max(ns, osizes[k]) for k, ns in nsizes.items()}
    over_hparams = {
        **{h: v for h, v in sampler_hparams.items() if h not in ["replacement"]},
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
    # combine and use sample(frac=1) to randomize the order of instances
    assert set(over_Xyy.iloc[:, -1].unique()).isdisjoint(under_Xyy.iloc[:, -1])
    resampled_Xyy = pd.concat([under_Xyy, over_Xyy], axis=0).sample(frac=1)
    resampled_X = resampled_Xyy.iloc[:, :-2]
    resampled_y = resampled_Xyy.iloc[:, -2]
    return resampled_X, resampled_y


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
        sampling_strategy="mixed",
        **sampler_hparams,
    ):
        self.fairness_info = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "unfavorable_labels": unfavorable_labels,
        }
        _validate_fairness_info(**self.fairness_info, check_schema=False)
        self.estimator = estimator
        self.redact = redact
        self.imbalance_repair_level = imbalance_repair_level
        self.bias_repair_level = bias_repair_level
        self.combine = combine
        self.sampling_strategy = sampling_strategy
        self.sampler_hparams = sampler_hparams

    @property
    def classes_(self):
        return self.estimator.classes_

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "not yet implemented"
        assert X.shape[0] == y.shape[0], (X.shape, y.shape)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index, name="y")
        diaeresis_y, fav_set = _make_diaeresis(X, y, self.fairness_info, self.combine)
        osizes = diaeresis_y.value_counts().sort_index().to_dict()
        nsizes = _orbis_pick_sizes(
            osizes,
            self.imbalance_repair_level,
            self.bias_repair_level,
            fav_set,
            self.sampling_strategy,
        )
        resampled_X, resampled_y = _orbis_resample(
            X, y, diaeresis_y, osizes, nsizes, self.sampler_hparams
        )
        if self.redact:
            self.redacting = Redacting(**self.fairness_info).fit(resampled_X)
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
            "required": [*_categorical_fairness_properties.keys(), "estimator"],
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
                "replacement": {
                    "description": "Whether under-sampling is with or without replacement.",
                    "type": "boolean",
                    "default": False,
                },
                "n_jobs": _hparam_n_jobs,
                "random_state": _hparam_random_state,
                "k_neighbors": {
                    **_hparam_n_neighbors,
                    "description": "Number of nearest neighbours to use to construct synthetic samples.",
                    "default": 5,
                },
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
    "description": """Experimental Orbis (Oversampling to Repair Bias and Imbalance Simultaneously) pre-estimator fairness mitigator.
Work in progress and subject to change; only supports pandas DataFrame so far.
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
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.orbis.html#lale.lib.aif360.orbis.Orbis",
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
