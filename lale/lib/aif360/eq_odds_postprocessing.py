# Copyright 2019 IBM Corporation
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

import aif360.algorithms.postprocessing

import lale.docstrings
import lale.operators

from .util import (
    _BasePostprocessingImpl,
    _dataset_fairness_properties,
    _numeric_input_predict_schema,
    _numeric_output_predict_schema,
    _numeric_supervised_input_fit_schema,
    _postprocessing_base_hyperparams,
)

_additional_hyperparams = {
    "unprivileged_groups": _dataset_fairness_properties["unprivileged_groups"],
    "privileged_groups": _dataset_fairness_properties["privileged_groups"],
    "seed": {
        "description": "Seed to make `predict` repeatable.",
        "anyOf": [{"type": "integer"}, {"enum": [None]}],
        "default": None,
    },
}


class EqOddsPostprocessingImpl(_BasePostprocessingImpl):
    def __init__(
        self,
        estimator,
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
        unprivileged_groups,
        privileged_groups,
        seed=None,
    ):
        mitigator = aif360.algorithms.postprocessing.EqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            seed=seed,
        )
        super(EqOddsPostprocessingImpl, self).__init__(
            mitigator=mitigator,
            estimator=estimator,
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
            protected_attribute_names=protected_attribute_names,
        )


_input_fit_schema = _numeric_supervised_input_fit_schema
_input_predict_schema = _numeric_input_predict_schema
_output_predict_schema = _numeric_output_predict_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": (
                list(_postprocessing_base_hyperparams.keys())
                + list(_additional_hyperparams.keys())
            ),
            "relevantToOptimizer": [],
            "properties": {
                **_postprocessing_base_hyperparams,
                **_additional_hyperparams,
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Equalized odds postprocessing`_ for fairness mitigation.

.. _`Equalized odds postprocessing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.eq_odds_postprocessing.html",
    "import_from": "aif360.algorithms.postprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier", "interpretable"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

lale.docstrings.set_docstrings(EqOddsPostprocessingImpl, _combined_schemas)

EqOddsPostprocessing = lale.operators.make_operator(
    EqOddsPostprocessingImpl, _combined_schemas
)
