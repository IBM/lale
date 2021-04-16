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

from .util import _BaseInEstimatorImpl, _categorical_fairness_properties


class _MetaFairClassifierImpl(_BaseInEstimatorImpl):
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        redact=True,
        preparation=None,
        tau=0.8,
        type="fdr",
    ):
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        mitigator = aif360.algorithms.inprocessing.MetaFairClassifier(
            tau=tau,
            sensitive_attr=prot_attr_names[0],
            type=type,
        )
        super(_MetaFairClassifierImpl, self).__init__(
            favorable_labels=favorable_labels,
            protected_attributes=protected_attributes,
            redact=redact,
            preparation=preparation,
            mitigator=mitigator,
        )


_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
    ],
}

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
                "tau",
                "type",
            ],
            "relevantToOptimizer": ["tau", "type"],
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
                "tau": {
                    "description": "Fairness penalty parameter.",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                },
                "type": {
                    "description": "The type of fairness metric to be used.",
                    "anyOf": [
                        {
                            "description": "False discovery rate ratio.",
                            "enum": ["fdr"],
                        },
                        {
                            "description": "Statistical rate / disparate impact.",
                            "enum": ["sr"],
                        },
                    ],
                    "default": "fdr",
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`MetaFairClassifier`_ in-estimator fairness mitigator. Takes the fairness metric as part of the input and returns a classifier optimized with respect to that fairness metric (`Celis et al. 2019`_).

.. _`MetaFairClassifier`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.MetaFairClassifier.html
.. _`Celis et al. 2019`: https://doi.org/10.1145/3287560.3287586
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.meta_fair_classifier.html#lale.lib.aif360.meta_fair_classifier.MetaFairClassifier",
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


MetaFairClassifier = lale.operators.make_operator(
    _MetaFairClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(MetaFairClassifier)
