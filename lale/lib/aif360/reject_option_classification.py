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


class _RejectOptionClassificationImpl(_BasePostEstimatorImpl):
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        estimator,
        redact=True,
        low_class_thresh=0.01,
        high_class_thresh=0.99,
        num_class_thresh=100,
        num_ROC_margin=50,
        metric_name="Statistical parity difference",
        metric_ub=0.05,
        metric_lb=-0.05,
    ):
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        privileged_groups = [{name: 1 for name in prot_attr_names}]
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
        super(_RejectOptionClassificationImpl, self).__init__(
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
                "low_class_thresh",
                "high_class_thresh",
                "num_class_thresh",
                "num_ROC_margin",
                "metric_name",
                "metric_ub",
                "metric_lb",
            ],
            "relevantToOptimizer": ["metric_name"],
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
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Reject option classification`_ post-estimator fairness mitigator. Gives favorable outcomes to unpriviliged groups and unfavorable outcomes to priviliged groups in a confidence band around the decision boundary with the highest uncertainty (`Kamiran et al. 2012`_).

.. _`Reject option classification`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.RejectOptionClassification.html
.. _`Kamiran et al. 2012`: https://doi.org/10.1109/ICDM.2012.45
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.reject_option_classification.html#lale.lib.aif360.reject_option_classification.RejectOptionClassification",
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


RejectOptionClassification = lale.operators.make_operator(
    _RejectOptionClassificationImpl, _combined_schemas
)

lale.docstrings.set_docstrings(RejectOptionClassification)
