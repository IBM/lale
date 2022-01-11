# Copyright  2020,2021 IBM Corporation
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

from aix360.algorithms.rbm import LogisticRuleRegression

import lale.docstrings
import lale.operators
from lale.lib.aif360.util import (
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)

"""This Class is used here to wrap the  Logistic Rule Regression"""


class _LogisticRuleRegressionImpl:
    def __init__(
        self,
        lambda0=0.05,
        dfTrainStd=None,
        dfTestStd=None,
        lambda1=0.01,
        useOrd=False,
        debias=True,
        init0=False,
        K=1,
        iterMax=200,
        B=1,
        wLB=0.5,
        stopEarly=False,
        eps=1e-6,
        maxSolverIter=100,
    ):
        self.mitigator = LogisticRuleRegression(
            lambda0=lambda0,
            lambda1=lambda1,
            useOrd=useOrd,
            debias=debias,
            init0=init0,
            K=K,
            iterMax=iterMax,
            B=B,
            wLB=wLB,
            stopEarly=stopEarly,
            eps=eps,
            maxSolverIter=maxSolverIter,
        )
        # Use standardized ordinal features
        self.useOrd = useOrd
        # Initialize with no features
        self.init0 = init0
        self.lambda0 = lambda0  # fixed cost of term
        self.lambda1 = lambda1  # cost per literal
        self.dfTrainStd = dfTrainStd
        self.dfTestStd = dfTestStd
        self.debias = debias  # re-fit final solution without regularization
        # Column generation parameters
        self.K = K  # maximum number of columns generated per iteration
        self.iterMax = iterMax  # maximum number of iterations
        self.B = B  # beam search width
        self.wLB = wLB  # weight on lower bound in evaluating nodes
        self.stopEarly = (
            stopEarly  # stop after current degree once improving column found
        )
        # Numerical tolerance on comparisons
        self.eps = eps
        # Maximum logistic solver iterations
        self.maxSolverIter = maxSolverIter

    def fit(self, dfTrain, yTrain):

        if self.dfTrainStd is not None:
            if len(self.dfTrainStd) > 0:
                return self.mitigator.fit(dfTrain, yTrain, self.dfTrainStd)
        else:
            return self.mitigator.fit(dfTrain, yTrain)

    def predict(self, dfTest):
        if type(self.dfTestStd) is not None:
            return self.mitigator.predict(dfTest, self.dfTestStd)
        else:
            return self.mitigator.predict(dfTest)

    def explain(self):
        return self.mitigator.explain()


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema
_input_predict_proba_schema = _categorical_input_predict_proba_schema
_output_predict_proba_schema = _categorical_output_predict_proba_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "lambda0",  # fixed cost of term
                "lambda1",  # cost per literal
                "dfTrainStd",
                "dfTestStd",
                "useOrd",
                "debias",  # re-fit final solution without regularization
                "init0",
                "K",  # maximum number of columns generated per iteration
                "iterMax",  # maximum number of iterations
                "B",  # beam search width
                "wLB",  # weight on lower bound in evaluating nodes
                "stopEarly",  # stop after current degree once improving column found
                "eps",  # Numerical tolerance on comparisons
                "maxSolverIter",  # Maximum logistic solver iterations
            ],
            "relevantToOptimizer": ["lambda0", "lambda1"],
            "properties": {
                "lambda0": {
                    "description": "Regularization - fixed cost of each rule.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 0.005,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "lambda1": {
                    "description": "Regularization - additional cost of each rule.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 0.001,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "dfTrainStd": {
                    "description": "Features; the outer array is over samples.",
                    "type": "array",
                    "default": None,
                },
                "dfTestStd": {
                    "description": "Features; the outer array is over samples.",
                    "type": "array",
                    "default": None,
                },
                "useOrd": {
                    "description": "use standardized numerical features",
                    "type": "boolean",
                    "default": False,
                },
                "debias": {
                    "description": "Re-fit final solution without regularization",
                    "type": "boolean",
                    "default": True,
                },
                "init0": {
                    "description": "Initialize with no features",
                    "type": "boolean",
                    "default": False,
                },
                "K": {
                    "description": "Column generation -maximum number of columns generated per iteration",
                    "type": "number",
                    "default": 1,
                },
                "iterMax": {
                    "description": "Column generation -maximum number of  iteration",
                    "type": "number",
                    "default": 200,
                },
                "B": {
                    "description": "Column generation - beam search width",
                    "type": "number",
                    "default": 1,
                },
                "wLB": {
                    "description": "Column generation -weight  on lower bound in evaluating nodes",
                    "type": "number",
                    "default": 0.5,
                },
                "stopEarly": {
                    "description": "Column generation -stop after current degree once  improving column found",
                    "type": "boolean",
                    "default": False,
                },
                "eps": {
                    "description": "Numerical tolerance on comparisons",
                    "type": "number",
                    "default": 1e-6,
                },
                "maxSolverIter": {
                    "description": "Maximum number of logistic regression solver iterations",
                    "type": "number",
                    "default": 100,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`logistic_rule_regression`_ :Logistic Rule Regression is a directly interpretable supervised learning method that performs logistic regression on rule-based features..

.. _`logistic_rule_regression`: https://aix360.readthedocs.io/en/latest/dise.html#aix360.algorithms.rbm.logistic_regression.LogisticRuleRegression

""",
    "import_from": "aix360.sklearn.inprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}

logisticruleregression = lale.operators.make_operator(
    _LogisticRuleRegressionImpl, _combined_schemas
)


lale.docstrings.set_docstrings(logisticruleregression)
