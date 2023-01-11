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
import lale.operators

from ._common_schemas import (
    _hparam_n_jobs,
    _hparam_n_neighbors,
    _hparam_operator,
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc,
    _input_fit_schema_cats,
    _input_predict_schema_cats,
    _input_transform_schema_cats,
    _output_decision_function_schema,
    _output_predict_proba_schema,
    _output_predict_schema,
    _output_transform_schema,
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
                "sampling_strategy": _hparam_sampling_strategy_anyof_neoc,
                "random_state": _hparam_random_state,
                "k_neighbors": {
                    **_hparam_n_neighbors,
                    "description": "Number of nearest neighbours to used to construct synthetic samples.",
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

lale.docstrings.set_docstrings(SMOTENC)
