# Copyright 2019-2023 IBM Corporation
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

import lale.docstrings
import lale.operators

from ._common_schemas import (
    _hparam_n_jobs,
    _hparam_n_neighbors,
    _hparam_operator,
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc,
    _input_fit_schema,
    _input_predict_schema,
    _input_transform_schema,
    _output_decision_function_schema,
    _output_predict_proba_schema,
    _output_predict_schema,
    _output_transform_schema,
)
from .base_resampler import _BaseResamplerImpl


class _SVMSMOTEImpl(_BaseResamplerImpl):
    def __init__(self, operator=None, **hyperparams):
        if operator is None:
            raise ValueError("Operator is a required argument.")

        resampler_instance = imblearn.over_sampling.SVMSMOTE(**hyperparams)
        super().__init__(operator=operator, resampler=resampler_instance)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": ["operator"],
            "relevantToOptimizer": ["operator"],
            "additionalProperties": False,
            "properties": {
                "operator": _hparam_operator,
                "sampling_strategy": _hparam_sampling_strategy_anyof_neoc,
                "random_state": _hparam_random_state,
                "k_neighbors": {
                    **_hparam_n_neighbors,
                    "description": "Number of nearest neighbours to used to construct synthetic samples.",
                    "default": 5,
                },
                "n_jobs": _hparam_n_jobs,
                "m_neighbors": {
                    **_hparam_n_neighbors,
                    "description": "Number of nearest neighbours to use to determine if a minority sample is in danger.",
                    "default": 10,
                },
                "svm_estimator": {
                    "description": "A parametrized sklearn.svm.SVC classifier can be passed.",
                    "anyOf": [{"laleType": "Any"}, {"enum": [None]}],
                    "default": None,
                },
                "out_step": {
                    "description": "Step size when extrapolating.",
                    "type": "number",
                    "default": 0.5,
                },
            },
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Over-sampling using SVM-SMOTE,
Variant of SMOTE algorithm which use an SVM algorithm to detect sample to use for generating new synthetic samples.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.imblearn.svm_smote.html",
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
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_predict_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}


SVMSMOTE = lale.operators.make_operator(_SVMSMOTEImpl, _combined_schemas)

lale.docstrings.set_docstrings(SVMSMOTE)
