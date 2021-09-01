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

import pandas as pd

import lale.docstrings
import lale.helpers
import lale.operators
import lale.pretty_print
from lale.lib.lale.hyperopt import Hyperopt
from lale.lib.sklearn import LogisticRegression

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class _OptimizeSuffix:

    _prefix: Optional[lale.operators.TrainedOperator]
    _optimizer: lale.operators.Operator

    def __init__(
        self,
        prefix: Optional[lale.operators.TrainedOperator] = None,
        suffix: Optional[lale.operators.Operator] = None,
        optimizer: Optional[lale.operators.PlannedIndividualOp] = None,
        optimizer_args=None,
        **kwargs
    ):
        self._prefix = prefix

        _suffix: lale.operators.Operator
        if suffix is None:
            _suffix = LogisticRegression()
        else:
            _suffix = suffix

        _optimizer: lale.operators.PlannedIndividualOp
        if optimizer is None:
            _optimizer = Hyperopt
        else:
            _optimizer = optimizer

        if optimizer_args is None:
            _optimizer_args = kwargs
        else:
            _optimizer_args = {**optimizer_args, **kwargs}

        self._optimizer = _optimizer(estimator=_suffix, **_optimizer_args)

    def fit(self, X_train, y_train=None, **kwargs):
        # Transform the input data using transformation steps in pipeline
        if self._prefix:
            X_train_transformed = self._prefix.transform(X_train)
            if isinstance(X_train, pd.DataFrame):
                X_train_transformed = pd.DataFrame(
                    data=X_train_transformed, index=X_train.index
                )
        else:
            X_train_transformed = X_train

        self._optimizer = self._optimizer.fit(X_train_transformed, y_train, **kwargs)
        return self

    def add_suffix(
        self, suffix: lale.operators.TrainedOperator
    ) -> lale.operators.TrainedOperator:
        trained: lale.operators.TrainedOperator
        """Given a trained suffix, adds it to the prefix to give a trained pipeline"""
        if self._prefix is None:
            trained = suffix
        else:
            trained = self._prefix >> suffix
        assert isinstance(trained, lale.operators.TrainedOperator)
        return trained

    def predict(self, X_eval, **predict_params):
        if self._prefix is None:
            input = X_eval
        else:
            input = self._prefix.transform(X_eval)
        return self._optimizer.predict(input, **predict_params)

    def summary(self, **kwargs):
        return self._optimizer.summary(**kwargs)

    def get_pipeline(self, pipeline_name=None, astype="lale", **kwargs):
        """Retrieve one of the trials.

        Parameters
        ----------
        pipeline_name : union type, default None

            - string
                Key for table returned by summary(), return a trainable pipeline.

            - None
                When not specified, return the best trained pipeline found.

        astype : 'lale' or 'sklearn', default 'lale'
            Type of resulting pipeline.

        Returns
        -------
        result : Trained operator if best, trainable operator otherwise."""
        result = self.add_suffix(
            self._optimizer.get_pipeline(
                pipeline_name=pipeline_name, astype=astype, **kwargs
            )
        )

        if result is None or astype == "lale":
            return result
        assert astype == "sklearn", astype
        return result.export_to_sklearn_pipeline()


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "prefix",
                "suffix",
                "optimizer",
            ],
            "relevantToOptimizer": [],
            "additionalProperties": True,
            "properties": {
                "prefix": {
                    "description": "Trained Lale operator or pipeline,\nby default None.",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "suffix": {
                    "description": "Lale operator or pipeline, which is to be optimized.\nIf (default) None is specified, LogisticRegression is used.",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "optimizer": {
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
    "description": """OptimizeSuffix is a wrapper around other optimizers, which runs the given optimizer
against the suffix, after transforming the data according to the prefix, and then stitches the result together into
a single trained pipeline.

Examples
--------
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.optimize_suffix.html",
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


OptimizeSuffix = lale.operators.make_operator(_OptimizeSuffix, _combined_schemas)

lale.docstrings.set_docstrings(OptimizeSuffix)
