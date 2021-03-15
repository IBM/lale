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

import copy
import logging
from typing import Any, Dict

from hyperopt import STATUS_OK

import lale.docstrings
import lale.operators
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import VotingClassifier

logger = logging.getLogger(__name__)


class _TopKVotingClassifierImpl:
    args_to_optimizer: Dict[str, Any]

    def __init__(
        self,
        estimator=None,
        optimizer=None,
        args_to_optimizer: Dict[str, Any] = None,
        k=10,
    ):
        self.estimator = estimator
        if self.estimator is None:
            raise ValueError("Estimator is a required argument.")
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = Hyperopt
        if args_to_optimizer is None:
            self.args_to_optimizer = {}
        else:
            self.args_to_optimizer = args_to_optimizer
        self.k = k

    def fit(self, X_train, y_train):
        optimizer_instance = self.optimizer(
            estimator=self.estimator, **self.args_to_optimizer
        )
        trained_optimizer1 = optimizer_instance.fit(X_train, y_train)
        results = trained_optimizer1.summary()
        results = results[
            results["status"] == STATUS_OK
        ]  # Consider only successful trials
        results = results.sort_values(by=["loss"], axis=0)
        k = min(self.k, results.shape[0])
        top_k_pipelines = results.iloc[0:k]
        pipeline_tuples = []
        for pipeline_name in top_k_pipelines.index:
            pipeline_instance = trained_optimizer1.get_pipeline(pipeline_name)
            pipeline_tuple = (pipeline_name, pipeline_instance)
            pipeline_tuples.append(pipeline_tuple)
        voting = VotingClassifier(estimators=pipeline_tuples)
        args_to_optimizer = copy.copy(self.args_to_optimizer)
        try:
            del args_to_optimizer["max_evals"]
        except KeyError:
            pass
        args_to_optimizer[
            "max_evals"
        ] = 1  # Currently, voting classifier has no useful hyperparameters to tune.
        optimizer_instance2 = self.optimizer(estimator=voting, **args_to_optimizer)
        trained_optimizer2 = optimizer_instance2.fit(X_train, y_train)
        self._best_estimator = trained_optimizer2.get_pipeline()
        return self

    def predict(self, X_eval):
        import warnings

        warnings.filterwarnings("ignore")
        if self._best_estimator is None:
            raise ValueError(
                "Can not predict as the best estimator is None. Either an attempt to call `predict` "
                "before calling `fit` or all the trials during `fit` failed."
            )
        trained = self._best_estimator
        try:
            predictions = trained.predict(X_eval)
        except ValueError as e:
            logger.warning(
                "ValueError in predicting using Hyperopt:{}, the error is:{}".format(
                    trained, e
                )
            )
            predictions = None
        return predictions

    def get_pipeline(self, pipeline_name=None, astype="lale"):
        """Retrieve one of the trials.

        Parameters
        ----------
        pipeline_name : None

        astype : 'lale' or 'sklearn', default 'lale'
            Type of resulting pipeline.

        Returns
        -------
        result : Trained operator if best, trainable operator otherwise."""
        assert pipeline_name is None
        result = self._best_estimator
        if result is None or astype == "lale":
            return result
        assert astype == "sklearn", astype
        assert isinstance(result, lale.operators.BasePipeline)
        return result.export_to_sklearn_pipeline()


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": ["estimator"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "estimator": {
                    "description": "Planned Lale individual operator or pipeline.",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "optimizer": {
                    "description": """Optimizer class to be used during the two stages of optimization.
                Default of None uses Hyperopt internally. Currently, only Hyperopt is supported as an optimizer.""",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "args_to_optimizer": {
                    "description": """Dictionary of keyword arguments required to be used for the given optimizer
                                as applicable for the given task. For example, max_evals, cv, scoring etc. for Hyperopt.
                                If None, default values for the optimizer would be used.""",
                    "anyOf": [
                        {"type": "object"},  # Python dictionary
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "k": {
                    "description": """Number of top pipelines to be used for the voting ensemble. If the number of
                            successful trials of the optimizer are less than k, the ensemble will use
                            only successful trials.""",
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
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
    "description": """This operator creates a voting ensemble from top k performing pipelines from the given planned pipeline.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.topk_voting_classifier.html",
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


TopKVotingClassifier = lale.operators.make_operator(
    _TopKVotingClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(TopKVotingClassifier)
