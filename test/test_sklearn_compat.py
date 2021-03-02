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

import unittest
from typing import Any, Dict

from sklearn.base import clone

from lale.lib.lale import ConcatFeatures as Concat
from lale.operators import make_operator


class _MutatingOpImpl:
    fit_counter: int
    predict_counter: int

    def __init__(self, k=0):
        self.fit_counter = 0
        self.predict_counter = 0
        self.k = k

    def fit(self, X, y=None):
        assert self.fit_counter == 0
        self.fit_counter = self.fit_counter + 1
        return self

    def predict(self, X, y=None):
        assert self.predict_counter == 0
        self.predict_counter = self.predict_counter + 1
        return [[1] for x in X]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["k"] = self.k
        return out

    def set_params(self, **impl_params):
        self.k = impl_params["k"]
        return self

    # def transform(self, X, y = None):
    #     return X, y


_input_schema_fit = {"$schema": "http://json-schema.org/draft-04/schema#"}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
}
_output_predict_schema = {"$schema": "http://json-schema.org/draft-04/schema#"}

_hyperparam_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {"k": {"type": "number"}},
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparam_schema,
        "input_fit": _input_schema_fit,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

MutatingOp = make_operator(_MutatingOpImpl, _combined_schemas)


def fit_clone_fit(op):
    op1 = op
    op1.fit(X=[1, 2], y=[1, 2])
    op2 = clone(op1)
    fit2 = op2.fit(X=[3, 4], y=[3, 4])
    print(fit2)


class TestClone(unittest.TestCase):
    def test_clone_clones_op(self):
        op = MutatingOp(k=1)
        fit_clone_fit(op)

    def test_clone_clones_seq(self):
        op = MutatingOp(k=1) >> MutatingOp(k=2)
        fit_clone_fit(op)

    def test_clone_clones_and(self):
        op = MutatingOp(k=1) & MutatingOp(k=2)
        fit_clone_fit(op)

    def test_clone_clones_concat(self):
        _ = ((MutatingOp(k=1) & MutatingOp(k=2))) >> Concat | MutatingOp(k=4)

    def test_clone_clones_choice(self):
        op = MutatingOp(k=1) | MutatingOp(k=2)
        fit_clone_fit(op)

    def test_clone_clones_complex(self):
        op = (
            (MutatingOp(k=1) | ((MutatingOp(k=2) & MutatingOp(k=3)) >> Concat))
            >> MutatingOp(k=4)
        ) | MutatingOp(k=5)
        fit_clone_fit(op)
