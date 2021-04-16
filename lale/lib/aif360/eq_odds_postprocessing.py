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
    _BasePostEstimatorImpl,
    _categorical_fairness_properties,
    _categorical_input_predict_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)


class _EqOddsPostprocessingImpl(_BasePostEstimatorImpl):
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        estimator,
        redact=True,
        seed=None,
    ):
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        privileged_groups = [{name: 1 for name in prot_attr_names}]
        mitigator = aif360.algorithms.postprocessing.EqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            seed=seed,
        )
        super(_EqOddsPostprocessingImpl, self).__init__(
            favorable_labels=favorable_labels,
            protected_attributes=protected_attributes,
            estimator=estimator,
            redact=redact,
            mitigator=mitigator,
        )


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "estimator",
                "redact",
                "seed",
            ],
            "relevantToOptimizer": [],
            "properties": {
                **_categorical_fairness_properties,
                "estimator": {
                    "description": "Nested supervised learning operator for which to mitigate fairness.",
                    "laleType": "operator",
                },
                "redact": {
                    "description": "Whether to redact protected attributes before data preparation (recommended) or not.",
                    "type": "boolean",
                    "default": True,
                },
                "seed": {
                    "description": "Seed to make `predict` repeatable.",
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                },
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Equalized odds postprocessing`_ post-estimator fairness mitigator. Solves a linear program to find probabilities with which to change output labels to optimize equalized odds (`Hardt et al. 2016`_, `Pleiss et al. 2017`_).

.. _`Equalized odds postprocessing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html
.. _`Hardt et al. 2016`: https://papers.nips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html
.. _`Pleiss et al. 2017`: https://proceedings.neurips.cc/paper/2017/hash/b8b9c74ac526fffbeb2d39ab038d1cd7-Abstract.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.eq_odds_postprocessing.html#lale.lib.aif360.eq_odds_postprocessing.EqOddsPostprocessing",
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

EqOddsPostprocessing = lale.operators.make_operator(
    _EqOddsPostprocessingImpl, _combined_schemas
)

lale.docstrings.set_docstrings(EqOddsPostprocessing)
