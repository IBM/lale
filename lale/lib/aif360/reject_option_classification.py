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
    _numeric_input_predict_schema,
    _numeric_output_predict_schema,
    _numeric_supervised_input_fit_schema,
    _postprocessing_base_hyperparams,
)

_additional_hyperparams = {
    "unprivileged_groups": {
        "description": "Representation for unprivileged group.",
        "type": "array",
        "items": {
            "description": "Map from feature names to group-indicating values.",
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "privileged_groups": {
        "description": "Representation for privileged group.",
        "type": "array",
        "items": {
            "description": "Map from feature names to group-indicating values.",
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "low_class_thresh": {
        "description": "Smallest classification threshold to use in the optimization.",
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": 0.01,
    },
    "high_class_thresh": {
        "description": "Highest classification threshold to use in the optimization.",
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": 0.99,
    },
    "num_class_thresh": {
        "description": "Number of classification thresholds between low_class_thresh and high_class_thresh for the optimization search.",
        "type": "integer",
        "minimum": 1,
        "default": 100,
    },
    "num_ROC_margin": {
        "description": "Number of relevant ROC margins to be used in the optimization search.",
        "type": "integer",
        "minimum": 1,
        "default": 50,
    },
    "metric_name": {
        "description": "Name of the metric to use for the optimization.",
        "enum": [
            "Statistical parity difference",
            "Average odds difference",
            "Equal opportunity difference",
        ],
        "default": "Statistical parity difference",
    },
    "metric_ub": {
        "description": "Upper bound of constraint on the metric value.",
        "type": "number",
        "default": 0.05,
    },
    "metric_lb": {
        "description": "Lower bound of constraint on the metric value.",
        "type": "number",
        "default": -0.05,
    },
}


class RejectOptionClassificationImpl(_BasePostprocessingImpl):
    def __init__(
        self,
        estimator,
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
        unprivileged_groups,
        privileged_groups,
        low_class_thresh=0.01,
        high_class_thresh=0.99,
        num_class_thresh=100,
        num_ROC_margin=50,
        metric_name="Statistical parity difference",
        metric_ub=0.05,
        metric_lb=-0.05,
    ):
        mitigator = aif360.algorithms.postprocessing.RejectOptionClassification(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            low_class_thresh=low_class_thresh,
            high_class_thresh=high_class_thresh,
            num_class_thresh=num_class_thresh,
            num_ROC_margin=num_ROC_margin,
            metric_name=metric_name,
            metric_ub=metric_ub,
            metric_lb=metric_lb,
        )
        super(RejectOptionClassificationImpl, self).__init__(
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
            "relevantToOptimizer": ["metric_name"],
            "properties": {
                **_postprocessing_base_hyperparams,
                **_additional_hyperparams,
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Reject option classification`_ postprocessing for fairness mitigation.

.. _`Reject option classification`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.RejectOptionClassification.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.reject_option_classification.html",
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

lale.docstrings.set_docstrings(RejectOptionClassificationImpl, _combined_schemas)

RejectOptionClassification = lale.operators.make_operator(
    RejectOptionClassificationImpl, _combined_schemas
)
