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
import aix360.algorithms.rbm.beam_search
import numpy as np
import pandas as pd
from aix360.algorithms.rbm import LogisticRuleRegression
from sklearn.linear_model import LogisticRegression

import lale
import lale.docstrings
import lale.operators
from lale.lib.aif360.util import (
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)

from .beam_search import beam_search, beam_search_K1

aix360.algorithms.rbm.beam_search.beam_search = beam_search
aix360.algorithms.rbm.beam_search.beam_search_K1 = beam_search_K1


def add_method(cls):
    """
    Decorator function to add/replace methods to existing classes.
    """

    def decorator(func):
        if type(cls) is list:
            for c in cls:
                setattr(c, func.__name__, func)
        else:
            setattr(cls, func.__name__, func)

    return decorator


@add_method(LogisticRuleRegression)
def fit(self, X, y, Xstd=None):
    """Fit model to training data.

    Args:
        X (DataFrame): Binarized features with MultiIndex column labels
        y (array): Target variable
        Xstd (DataFrame, optional): Standardized numerical features
    Returns:
        LogisticRuleRegression: Self
    """
    # Initialization
    # Number of samples
    numOrd = None
    Astd = None
    # B = None
    n = X.shape[0]
    if self.init0:
        # Initialize with empty feature indicator and conjunction matrices
        z = pd.DataFrame([], index=X.columns)
        A = np.empty((X.shape[0], 0))
    else:
        # Initialize with X itself i.e. singleton conjunctions
        # Feature indicator and conjunction matrices
        z = pd.DataFrame(np.eye(X.shape[1], dtype=int), index=X.columns)
        # Remove negations
        indPos = X.columns.get_level_values(1).isin(["", "<=", "=="])
        z = z.loc[:, indPos]
        A = X.loc[:, indPos].values
        # Scale conjunction matrix to account for non-uniform penalties
        A = A * self.lambda0 / (self.lambda0 + self.lambda1 * z.sum().values)
    if self.useOrd:
        self.namesOrd = Xstd.columns
        numOrd = Xstd.shape[1]
        # Scale ordinal features to have similar std as "average" binary feature
        Astd = 0.4 * Xstd.values
    # Iteration counter
    self.it = 0
    # Logistic regression object
    lr = LogisticRegression(
        penalty="l1",
        C=1 / (n * self.lambda0),
        solver="saga",
        multi_class="ovr",
        max_iter=self.maxSolverIter,
    )

    self.p = y.mean()
    if self.init0:
        # Initial residual
        r = (self.p - y) / n
        # Derivative w.r.t. intercept term
        UB = min(r.sum(), 0)
    else:
        # Fit logistic regression model
        if self.useOrd:
            B = np.concatenate((Astd, A), axis=1)
            lr.fit(B, y)
            # Initial residual
            r = (lr.predict_proba(B)[:, 1] - y) / n
        else:
            lr.fit(A, y)
            # Initial residual
            r = (lr.predict_proba(A)[:, 1] - y) / n
        # Most "negative" subderivative among current variables (undo scaling)
        UB = -np.abs(np.dot(r, A))
        UB *= (self.lambda0 + self.lambda1 * z.sum().values) / self.lambda0
        UB += self.lambda0 + self.lambda1 * z.sum().values
        UB = min(UB.min(), 0)

    # Beam search for conjunctions with subdifferentials that exclude zero
    vp, zp, Ap = beam_search_K1(
        r,
        X,
        self.lambda0,
        self.lambda1,
        UB=UB,
        B=self.B,
        wLB=self.wLB,
        eps=self.eps,
        stopEarly=self.stopEarly,
    )
    vn, zn, An = beam_search_K1(
        -r,
        X,
        self.lambda0,
        self.lambda1,
        UB=UB,
        B=self.B,
        wLB=self.wLB,
        eps=self.eps,
        stopEarly=self.stopEarly,
    )
    v = np.append(vp, vn)

    while (v < UB).any() and (self.it < self.iterMax):
        # Subdifferentials excluding zero exist, continue
        self.it += 1
        zNew = pd.concat([zp, zn], axis=1, ignore_index=True)
        Anew = np.concatenate((Ap, An), axis=1)

        # K conjunctions with largest subderivatives in absolute value
        idxLargest = np.argsort(v)[: self.K]
        v = v[idxLargest]
        zNew = zNew.iloc[:, idxLargest]
        Anew = Anew[:, idxLargest]
        # Scale new conjunction matrix to account for non-uniform penalties
        Anew = Anew * self.lambda0 / (self.lambda0 + self.lambda1 * zNew.sum().values)

        # Add to existing conjunctions
        z = pd.concat([z, zNew], axis=1, ignore_index=True)
        A = np.concatenate((A, Anew), axis=1)
        # Fit logistic regression model
        if self.useOrd:
            B = np.concatenate((Astd, A), axis=1)
            lr.fit(B, y)
            # Residual
            r = (lr.predict_proba(B)[:, 1] - y) / n
        else:
            lr.fit(A, y)
            # Residual
            r = (lr.predict_proba(A)[:, 1] - y) / n

        # Most "negative" subderivative among current variables (undo scaling)
        UB = -np.abs(np.dot(r, A))
        UB *= (self.lambda0 + self.lambda1 * z.sum().values) / self.lambda0
        UB += self.lambda0 + self.lambda1 * z.sum().values
        UB = min(UB.min(), 0)

        # Beam search for conjunctions with subdifferentials that exclude zero
        vp, zp, Ap = beam_search_K1(
            r,
            X,
            self.lambda0,
            self.lambda1,
            UB=UB,
            B=self.B,
            wLB=self.wLB,
            eps=self.eps,
            stopEarly=self.stopEarly,
        )
        vn, zn, An = beam_search_K1(
            -r,
            X,
            self.lambda0,
            self.lambda1,
            UB=UB,
            B=self.B,
            wLB=self.wLB,
            eps=self.eps,
            stopEarly=self.stopEarly,
        )
        v = np.append(vp, vn)

    # Restrict model to conjunctions with nonzero coefficients
    try:
        idxNonzero = np.where(np.abs(lr.coef_) > self.eps)[1]
        if self.useOrd:
            # Nonzero indices of standardized and rule features
            self.idxNonzeroOrd = idxNonzero[idxNonzero < numOrd]
            nnzOrd = len(self.idxNonzeroOrd)
            idxNonzeroRules = idxNonzero[idxNonzero >= numOrd] - numOrd
            B = 0.0
            if self.debias and len(idxNonzero):
                # Re-fit logistic regression model with effectively no regularization
                z = z.iloc[:, idxNonzeroRules]
                lr.C = 1 / self.eps
                lr.fit(B[:, idxNonzero], y)
                idxNonzero = np.where(np.abs(lr.coef_) > self.eps)[1]
                # Nonzero indices of standardized and rule features
                idxNonzeroOrd2 = idxNonzero[idxNonzero < nnzOrd]
                self.idxNonzeroOrd = self.idxNonzeroOrd[idxNonzeroOrd2]
                idxNonzeroRules = idxNonzero[idxNonzero >= nnzOrd] - nnzOrd
            self.z = z.iloc[:, idxNonzeroRules]
            lr.coef_ = lr.coef_[:, idxNonzero]
        else:
            if self.debias and len(idxNonzero):
                # Re-fit logistic regression model with effectively no regularization
                z = z.iloc[:, idxNonzero]
                lr.C = 1 / self.eps
                lr.fit(A[:, idxNonzero], y)
                idxNonzero = np.where(np.abs(lr.coef_) > self.eps)[1]
            self.z = z.iloc[:, idxNonzero]
            lr.coef_ = lr.coef_[:, idxNonzero]
    except AttributeError:
        # Model has no coefficients except intercept
        self.z = z
    self.lr = lr


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
