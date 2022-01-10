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

import logging
from typing import Any, Dict, Optional

import lale.docstrings
import lale.operators
import lale.pretty_print
from lale.lib.lale.optimize_suffix import OptimizeSuffix

from ._common_schemas import schema_estimator

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class _OptimizeLast:

    _suffix_optimizer: lale.operators.Operator

    def __init__(
        self,
        estimator: Optional[lale.operators.TrainedOperator] = None,
        last_optimizer: Optional[lale.operators.Operator] = None,
        optimizer_args=None,
        **kwargs
    ):
        if estimator is None:
            last_estimator = None
            lale_prefix = None
        elif isinstance(estimator, lale.operators.TrainedIndividualOp):
            lale_prefix = None
            last_estimator = estimator.clone()
        else:
            assert isinstance(estimator, lale.operators.TrainedPipeline)
            steps = estimator.steps_list()
            num_steps = len(steps)
            if num_steps == 0:
                last_estimator = None
            else:
                last_estimator = estimator.steps_list()[-1].clone()
            lale_prefix = estimator.remove_last()

        self._suffix_optimizer = OptimizeSuffix(
            prefix=lale_prefix,
            suffix=last_estimator,
            optimizer=last_optimizer,
            optimizer_args=optimizer_args,
            **kwargs
        )

    def __getattr__(self, item):
        return getattr(self._suffix_optimizer.shallow_impl, item)

    def fit(self, X_train, y_train=None, **kwargs):
        return self._suffix_optimizer.fit(X_train, y_train, **kwargs)

    def predict(self, X_eval, **predict_params):
        return self._suffix_optimizer.predict(X_eval, **predict_params)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimator",
                "last_optimizer",
            ],
            "relevantToOptimizer": [],
            "additionalProperties": True,
            "properties": {
                "estimator": schema_estimator,
                "last_optimizer": {
                    "description": "Lale optimizer.\nIf (default) None is specified, Hyperopt is used.",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "optimizer_args": {
                    "description": "Parameters to be passed to the optimizer",
                    "anyOf": [
                        {"type": "object"},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {"X": {}, "y": {}},
}

_input_predict_schema = {"type": "object", "required": ["X"], "properties": {"X": {}}}

_output_predict_schema: Dict[str, Any] = {}

_combined_schemas = {
    "description": """OptimizeLast is a wrapper around other optimizers, which runs the given optimizer
against the suffix, after transforming the data according to the prefix, and then stitches the result together into
a single trained pipeline.

Examples
--------
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.optimize_last.html",
    "import_from": "lale.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


OptimizeLast = lale.operators.make_operator(_OptimizeLast, _combined_schemas)

lale.docstrings.set_docstrings(OptimizeLast)
