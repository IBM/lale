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
import sklearn.linear_model

import lale.docstrings
import lale.operators

from .util import (
    _BaseInEstimatorImpl,
    _categorical_fairness_properties,
    _categorical_input_predict_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)


class _GerryFairClassifierImpl(_BaseInEstimatorImpl):
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        redact=True,
        preparation=None,
        C=10,
        printflag=False,
        heatmapflag=False,
        heatmap_iter=10,
        heatmap_path=".",
        max_iters=10,
        gamma=0.01,
        fairness_def="FP",
        predictor=None,
    ):
        if predictor is None:
            predictor = sklearn.linear_model.LinearRegression()
        if isinstance(predictor, lale.operators.Operator):
            if isinstance(predictor, lale.operators.IndividualOp):
                predictor = predictor._impl_instance()._wrapped_model
            else:
                raise ValueError(
                    "If predictor is a Lale operator, it needs to be an individual operator."
                )
        mitigator = aif360.algorithms.inprocessing.GerryFairClassifier(
            C=C,
            printflag=printflag,
            heatmapflag=heatmapflag,
            heatmap_iter=heatmap_iter,
            heatmap_path=heatmap_path,
            max_iters=max_iters,
            gamma=gamma,
            fairness_def=fairness_def,
            predictor=predictor,
        )
        super(_GerryFairClassifierImpl, self).__init__(
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
                "C",
                "printflag",
                "heatmapflag",
                "heatmap_iter",
                "heatmap_path",
                "max_iters",
                "gamma",
                "fairness_def",
                "predictor",
            ],
            "relevantToOptimizer": ["C", "max_iters", "gamma", "fairness_def"],
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
                "C": {
                    "description": "Maximum L1 norm for the dual variables.",
                    "type": "number",
                    "default": 10,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "printflag": {
                    "description": "Print output flag.",
                    "type": "boolean",
                    "default": False,
                },
                "heatmapflag": {
                    "description": "Save heatmaps every heatmap_iter flag.",
                    "type": "boolean",
                    "default": False,
                },
                "heatmap_iter": {
                    "description": "Save heatmaps every heatmap_iter.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                },
                "heatmap_path": {
                    "description": "Save heatmaps path.",
                    "type": "string",
                    "default": ".",
                },
                "max_iters": {
                    "description": "Time horizon for the fictitious play dynamic.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                    "distribution": "loguniform",
                    "maximumForOptimizer": 1000,
                },
                "gamma": {
                    "description": "Fairness approximation parameter.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 0.01,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 0.001,
                    "maximumForOptimizer": 1.0,
                },
                "fairness_def": {
                    "description": "Fairness notion.",
                    "enum": ["FP", "FN"],
                    "default": "FP",
                },
                "predictor": {
                    "description": "Hypothesis class for the learner.",
                    "anyOf": [
                        {
                            "description": "Supports LR, SVM, KR, Trees.",
                            "laleType": "operator",
                        },
                        {
                            "description": "sklearn.linear_model.LinearRegression",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`GerryFairClassifier`_ in-estimator fairness mitigator. Attempts to learn classifiers that are fair with respect to rich subgroups (`Kearns et al. 2018`_, `Kearns et al. 2019`_). Rich subgroups are defined by (linear) functions over the sensitive attributes, and fairness notions are statistical: false positive, false negative, and statistical parity rates. This implementation uses a max of two regressions as a cost-sensitive classification oracle, and supports linear regression, support vector machines, decision trees, and kernel regression.

.. _`GerryFairClassifier`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.GerryFairClassifier.html
.. _`Kearns et al. 2018`: http://proceedings.mlr.press/v80/kearns18a.html
.. _`Kearns et al. 2019`: https://doi.org/10.1145/3287560.3287592
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.gerry_fair_classifier.html#lale.lib.aif360.gerry_fair_classifier.GerryFairClassifier",
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

GerryFairClassifier = lale.operators.make_operator(
    _GerryFairClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(GerryFairClassifier)
