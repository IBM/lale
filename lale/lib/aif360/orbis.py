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


def _assert_almost_equal(v1, v2):
    assert abs(v1 - v2) < 0.00001, (v1, v2)


def _class_imbalance(s00, s01, s10, s11):
    return (s00 + s10) / (s01 + s11)


def _disparate_impact(s00, s01, s10, s11):
    return (s01 / (s00 + s01)) / (s11 / (s10 + s11))


def _sizes_to_string(sizes, prefix):
    keys = sorted(sizes.keys())
    sizes_string = ", ".join(f"{prefix}{k} {sizes[k]:3d}" for k in keys)
    ci_string = f"{prefix}ci {_class_imbalance(*(sizes[k] for k in keys)):.3f}"
    di_string = f"{prefix}di {_disparate_impact(*(sizes[k] for k in keys)):.3f}"
    return f"{sizes_string}, {ci_string}, {di_string}"


def _mapping_is_invertible(mapping):
    return (
        set(mapping.keys()) == {"00", "01", "10", "11"}
        and set(mapping.values()) == {"00", "01", "10", "11"}
        and all(mapping[mapping[k]] == k for k in mapping)
    )


def _pick_sizes_assuming_oci_and_odi_at_most_one(
    osizes: Dict[str, int], imbalance_repair_level: float, bias_repair_level: float
) -> Dict[str, int]:
    """Pick new sizes for each intersection assuming the old 11 size is largest.
    If some other size is largest, use wrapper _pick_sizes_symmetric instead.

    Parameters
    ----------
    osizes : dictionary from string to integer
        Maps intersection names ["00", "01", "10", "11"] to their old sizes.
        The first subscript is the group (0 unprivileged, 1 privileged) and
        the second subscript is the class (0 unfavorable, 1 favorable).

    imbalance_repair_level : number, >= 0, <= 1
        How much to repair for class imbalance, where
        0 means original imbalance and 1 means perfect balance.

    bias_repair_level : number, >= 0, <= 1
        How much to repair for group bias, where
        0 means original bias and 1 means perfect fairness.

    Returns
    -------
    nsizes : dictionary from string to integer
        Maps intersection names to their new sizes with the same keys as osizes.
    """
    # inputs
    o00, o01, o10, o11 = osizes["00"], osizes["01"], osizes["10"], osizes["11"]
    # outputs: new intersection sizes n00, n01, n10, n11
    # constants
    oci = _class_imbalance(o00, o01, o10, o11)
    nci = oci + imbalance_repair_level * (1 - oci)
    odi = _disparate_impact(o00, o01, o10, o11)
    ndi = odi + bias_repair_level * (1 - odi)
    # we have two equations, one each for nci and ndi
    #    nci == (n00 + n10) / (n01 + n11)
    #    ndi == (n01 / (n00 + n01)) / (n11 / (n10 + n11))
    # without loss of generality, assume oci <= 1 and odi <= 1
    assert oci <= 1 and odi <= 1, _sizes_to_string(osizes, "o")
    # that means we do not need to upsample o11
    # we will set n11 == o11, leaving three unknowns: n00, n01, n10
    # two equations admit multiple solutions for the three unknowns
    # algorithm to pick the solution that minimizes the amount of oversampling:
    # - loop over candidate values for n00 in ascending order
    # - given n00, solve the equations to also pick values for n01 and n10
    # - terminate when all group sizes >= their original size

    invalid = {"00": -1, "01": -1, "10": -1, "11": -1}

    def solve_for_n01_n10_given_00_11(n00, n11):
        # rewriting the nci equation:
        #    n01 == n00 / nci + n10 / nci - n11
        # rewriting the ndi equation:
        #    ndi * n11 * n00 + ndi * n11 * n01 == n01 * n10 + n01 * n11
        # substituting n01 into the ndi equation:
        #    ndi*n11*n00 + ndi*n11*(n00/nci + n10/nci - n11)
        #    == n10*(n00/nci + n10/nci - n11) + n11*(n00/nci + n10/nci - n11)
        # rewriting this to the standard form of a quadratic equation for n10:
        #    n10*n10
        #      + n10*(n00 + n11 - n11*nci - ndi*n11)
        #      + (n11*n00+ndi*n11*n11*nci-n11*n11*nci-ndi*n11*n00*nci-ndi*n11*n00)
        #    == 0
        # assigning variables so the above is n10*n10 + n10 * b + c == 0:
        b = n00 + n11 - n11 * nci - ndi * n11
        c = (
            n11 * n00
            + ndi * n11 * n11 * nci
            - n11 * n11 * nci
            - ndi * n11 * n00 * nci
            - ndi * n11 * n00
        )
        # the square root of a negative number is imaginary
        if b * b - 4 * c < 0:
            return invalid
        # quadratic equations have two solutions
        n10_plus = (-b + (b * b - 4 * c) ** 0.5) / 2
        n01_plus = n00 / nci + n10_plus / nci - n11
        valid_plus = round(n01_plus) >= o01 and round(n10_plus) >= o10
        n10_minus = (-b - (b * b - 4 * c) ** 0.5) / 2
        n01_minus = n00 / nci + n10_minus / nci - n11
        valid_minus = round(n01_minus) >= o01 and round(n10_minus) >= o10
        if valid_plus and valid_minus:  # pick solution minimizing n01 + n10
            if n01_plus + n10_plus < n01_minus + n01_minus:
                n01, n10 = n01_plus, n10_plus
            else:
                n01, n10 = n01_minus, n10_minus
        elif valid_plus:
            n01, n10 = n01_plus, n10_plus
        elif valid_minus:
            n01, n10 = n01_minus, n10_minus
        else:
            return invalid
        _assert_almost_equal(nci, _class_imbalance(n00, n01, n10, n11))
        _assert_almost_equal(ndi, _disparate_impact(n00, n01, n10, n11))
        return {"00": n00, "01": round(n01), "10": round(n10), "11": n11}

    nsizes = invalid
    # to minimize n00, search candidate values in ascending order
    for n00 in range(o00, sum(osizes.values()) + 1):
        nsizes = solve_for_n01_n10_given_00_11(n00, o11)
        okay = all(nsizes[k] >= osizes[k] for k in osizes)
        if okay:
            break
    if not all(nsizes[k] >= osizes[k] for k in osizes):
        logger.warning(f"insufficient upsampling for {osizes}")
        nsizes = {
            "00": max(osizes["00"], osizes["01"]),
            "01": max(osizes["00"], osizes["01"]),
            "10": max(osizes["10"], osizes["11"]),
            "11": max(osizes["10"], osizes["11"]),
        }
    return nsizes


def _pick_sizes_assuming_oci_at_most_one(
    osizes: Dict[str, int], imbalance_repair_level: float, bias_repair_level: float
) -> Dict[str, int]:
    oci = _class_imbalance(osizes["00"], osizes["01"], osizes["10"], osizes["11"])
    assert oci <= 1, _sizes_to_string(osizes, "o")
    odi = _disparate_impact(osizes["00"], osizes["01"], osizes["10"], osizes["11"])
    if odi <= 1:
        mapping = {"00": "00", "01": "01", "10": "10", "11": "11"}  # identity
    else:
        mapping = {"00": "10", "01": "11", "10": "00", "11": "01"}  # swap groups
    assert _mapping_is_invertible(mapping), mapping
    mapped_osizes = {k1: osizes[k2] for k1, k2 in mapping.items()}
    mapped_nsizes = _pick_sizes_assuming_oci_and_odi_at_most_one(
        mapped_osizes, imbalance_repair_level, bias_repair_level
    )
    nsizes = {k1: mapped_nsizes[k2] for k1, k2 in mapping.items()}
    return nsizes


def _pick_sizes(
    osizes: Dict[str, int], imbalance_repair_level: float, bias_repair_level: float, favorable_labels: Set[int]
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
                    set(self.favorable_labels),
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
