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

import typing

import imblearn.over_sampling
import numpy as np
from packaging import version

import lale.docstrings
import lale.operators

from ._common_schemas import (
    _hparam_categorical_encoder,
    _hparam_n_jobs,
    _hparam_n_neighbors,
    _hparam_operator,
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc_over,
    _input_fit_schema_cats,
    _input_predict_schema_cats,
    _input_transform_schema_cats,
    _output_decision_function_schema,
    _output_predict_proba_schema,
    _output_predict_schema,
    _output_transform_schema,
    imblearn_version,
)
from .base_resampler import _BaseResamplerImpl


class _SMOTENCImpl(_BaseResamplerImpl):
    def __init__(self, operator=None, **hyperparams):
        if operator is None:
            raise ValueError("Operator is a required argument.")
        self._hyperparams = hyperparams

        super().__init__(operator=operator, resampler=None)

    def fit(self, X, y=None):
        if self.resampler is None:
            if self._hyperparams["categorical_features"] is None:
                self._hyperparams["categorical_features"] = [
                    not np.issubdtype(typ, np.number) for typ in X.dtypes
                ]
            self.resampler = imblearn.over_sampling.SMOTENC(**self._hyperparams)
        return super().fit(X, y)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": ["operator"],
            "relevantToOptimizer": ["operator"],
            "additionalProperties": False,
            "properties": {
                "operator": _hparam_operator,
                "categorical_features": {
                    "description": "Specifies which features are categorical.",
                    "anyOf": [
                        {
                            "description": "Treat all features with non-numeric dtype as categorical.",
                            "enum": [None],
                        },
                        {
                            "description": "Indices specifying the categorical features.",
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        {
                            "description": "Mask array of shape `(n_features,)` where True indicates the categorical features.",
                            "type": "array",
                            "items": {"type": "boolean"},
                        },
                    ],
                    "default": None,
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
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC).
Can handle some nominal features, but not designed to work with only nominal features.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.imblearn.smotenc.html",
    "import_from": "imblearn.over_sampling",
    "type": "object",
    "tags": {
        "pre": [],
        "op": [
            "transformer",
            "estimator",
            "resampler",
        ],  # transformer and estimator both as a higher-order operator
        "post": [],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema_cats,
        "input_transform": _input_transform_schema_cats,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema_cats,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_schema_cats,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_predict_schema_cats,
        "output_decision_function": _output_decision_function_schema,
    },
}


SMOTENC = lale.operators.make_operator(_SMOTENCImpl, _combined_schemas)

if imblearn_version is not None and imblearn_version >= version.Version("0.11"):
    SMOTENC = typing.cast(
        lale.operators.PlannedIndividualOp,
        SMOTENC.customize_schema(
            n_jobs=None,
            categorical_encoder=_hparam_categorical_encoder,
            categorical_features={
                "description": "Specifies which features are categorical.",
                "default": "auto",
                "anyOf": [
                    {
                        "description": "Treat all features with non-numeric dtype as categorical.",
                        "enum": [None],
                    },
                    {
                        "description": "Automatically detect categorical features. Only supported when X is a pandas.DataFrame and it corresponds to columns that have a pandas.CategoricalDtype.",
                        "enum": ["auto"],
                    },
                    {
                        "description": "Indices specifying the categorical features.",
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    {
                        "description": "Array of str corresponding to the feature names. X should be a pandas pandas.DataFrame in this case.",
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    {
                        "description": "Mask array of shape `(n_features,)` where True indicates the categorical features.",
                        "type": "array",
                        "items": {"type": "boolean"},
                    },
                ],
            },
            set_as_available=True,
        ),
    )

if imblearn_version is not None and imblearn_version >= version.Version("0.12"):
    SMOTENC = typing.cast(
        lale.operators.PlannedIndividualOp,
        SMOTENC.customize_schema(
            n_jobs=None,
            set_as_available=True,
        ),
    )

lale.docstrings.set_docstrings(SMOTENC)
