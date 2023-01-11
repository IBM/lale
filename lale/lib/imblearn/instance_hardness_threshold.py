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

import imblearn.under_sampling

import lale.docstrings
import lale.operators

from ._common_schemas import (
    _hparam_n_jobs,
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


class _InstanceHardnessThresholdImpl(_BaseResamplerImpl):
    def __init__(self, operator=None, **hyperparams):
        if operator is None:
            raise ValueError("Operator is a required argument.")

        resampler_instance = imblearn.under_sampling.InstanceHardnessThreshold(
            **hyperparams
        )
        super().__init__(operator=operator, resampler=resampler_instance)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "relevantToOptimizer": ["operator"],
            "additionalProperties": False,
            "properties": {
                "operator": _hparam_operator,
                "estimator": {
                    "description": """Classifier to be used to estimate instance hardness of the samples.
By default a :class:`sklearn.ensemble.RandomForestClassifer` will be used.
If ``str``, the choices using a string are the following: ``'knn'``,
``'decision-tree'``, ``'random-forest'``, ``'adaboost'``,
``'gradient-boosting'`` and ``'linear-svm'``.  If object, an estimator
inherited from :class:`sklearn.base.ClassifierMixin` and having an
attribute :func:`predict_proba`.""",
                    "anyOf": [
                        {"laleType": "Any"},
                        {
                            "enum": [
                                "knn",
                                "decision-tree",
                                "random-forest",
                                "adaboost",
                                "gradient-boosting",
                                "linear-svm",
                            ]
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "sampling_strategy": _hparam_sampling_strategy_anyof_neoc,
                "random_state": _hparam_random_state,
                "cv": {
                    "description": "Number of folds to be used when estimating samples’ instance hardness.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                },
                "n_jobs": _hparam_n_jobs,
            },
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Class to perform under-sampling based on the instance hardness threshold.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.imblearn.instance_hardness_threshold.html",
    "import_from": "imblearn.under_sampling",
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


InstanceHardnessThreshold = lale.operators.make_operator(
    _InstanceHardnessThresholdImpl, _combined_schemas
)

lale.docstrings.set_docstrings(InstanceHardnessThreshold)
