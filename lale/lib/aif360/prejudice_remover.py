# Copyright 2020 IBM Corporation
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

import aif360.algorithms.inprocessing

import lale.docstrings
import lale.operators

from .util import (
    _BaseInEstimatorImpl,
    _categorical_fairness_properties,
    _categorical_input_predict_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)


class _PrejudiceRemoverImpl(_BaseInEstimatorImpl):
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        redact=True,
        preparation=None,
        eta=1.0,
    ):
        mitigator = aif360.algorithms.inprocessing.PrejudiceRemover(eta=eta)
        super(_PrejudiceRemoverImpl, self).__init__(
            favorable_labels=favorable_labels,
            protected_attributes=protected_attributes,
            redact=redact,
            preparation=preparation,
            mitigator=mitigator,
        )


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "redact",
                "preparation",
                "eta",
            ],
            "relevantToOptimizer": ["eta"],
            "properties": {
                **_categorical_fairness_properties,
                "redact": {
                    "description": "Whether to redact protected attributes before data preparation (recommended) or not.",
                    "type": "boolean",
                    "default": True,
                },
                "preparation": {
                    "description": "Transformer, which may be an individual operator or a sub-pipeline.",
                    "anyOf": [
                        {"laleType": "operator"},
                        {"description": "lale.lib.lale.NoOp", "enum": [None]},
                    ],
                    "default": None,
                },
                "eta": {
                    "description": "Fairness penalty parameter.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`PrejudiceRemover`_ in-estimator fairness mitigator. Adds a discrimination-aware regularization term to the learning objective (`Kamishima et al. 2012`_).

.. _`PrejudiceRemover`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html
.. _`Kamishima et al. 2012`: https://doi.org/10.1007/978-3-642-33486-3_3
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.prejudice_remover.html#lale.lib.aif360.prejudice_remover.PrejudiceRemover",
    "import_from": "aif360.sklearn.inprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


PrejudiceRemover = lale.operators.make_operator(
    _PrejudiceRemoverImpl, _combined_schemas
)

lale.docstrings.set_docstrings(PrejudiceRemover)
