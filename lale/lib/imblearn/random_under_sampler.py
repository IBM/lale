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

import imblearn.under_sampling

import lale.docstrings
import lale.operators

from ._common_schemas import (
    _hparam_operator,
    _hparam_random_state,
    _hparam_sampling_strategy_anyof_neoc_under,
    _input_fit_schema_cats,
    _input_predict_schema_cats,
    _input_transform_schema_cats,
    _output_decision_function_schema,
    _output_predict_proba_schema,
    _output_predict_schema,
    _output_transform_schema,
)
from .base_resampler import _BaseResamplerImpl


class _RandomUnderSamplerImpl(_BaseResamplerImpl):
    def __init__(self, *, operator=None, **hyperparams):
        if operator is None:
            raise ValueError("Operator is a required argument.")

        resampler_instance = imblearn.under_sampling.RandomUnderSampler(**hyperparams)
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
                "sampling_strategy": _hparam_sampling_strategy_anyof_neoc_under,
                "random_state": _hparam_random_state,
                "replacement": {
                    "description": "Whether the sample is with or without replacement.",
                    "type": "boolean",
                    "default": False,
                },
            },
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Class to perform random under-sampling, i.e. under-sample the minority class(es) by picking samples at random with replacement.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.imblearn.random_under_sampler.html",
    "import_from": "imblearn.under_sampling",
    "type": "object",
    "tags": {
        "pre": [],
        "op": [
            "transformer",
            "estimator",
            "resampler",
            "classifier",
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


RandomUnderSampler = lale.operators.make_operator(
    _RandomUnderSamplerImpl, _combined_schemas
)

lale.docstrings.set_docstrings(RandomUnderSampler)
