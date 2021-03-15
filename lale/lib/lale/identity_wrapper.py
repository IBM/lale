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

import lale.docstrings
import lale.operators


class _IdentityWrapperImpl:
    # This should be equivalent to:
    # the underlying operator:
    # IdentityWrapper(op) should behave the same as op
    def __init__(self, op=None):
        self._hyperparams = {"op": op}

    def getOp(self):
        return self._hyperparams["op"]

    def transform(self, X, y=None):
        return self.getOp().transform(X, y=y)

    def transform_schema(self, s_X):
        return self.getOp().transform_schema(s_X)

    def input_schema_fit(self):
        return self.getOp().input_schema_fit()

    def predict(self, X):
        return self.getOp().predict(X)

    def predict_proba(self, X):
        return self.getOp().predict_proba(self, X)

    def fit(self, X, y=None):
        return self.getOp().fit(X, y=y)

    # def get_feature_names(self, input_features=None):
    #     if input_features is not None:
    #         return list(input_features)
    #     elif self._feature_names is not None:
    #         return self._feature_names
    #     else:
    #         raise ValueError('Can only call get_feature_names on a trained operator. Please call fit to get a trained operator.')


_hyperparams_schema = {
    "description": "Hyperparameter schema for the identity Higher Order Operator, which wraps another operator and runs it as usual",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": ["op"],
            "properties": {"op": {"laleType": "operator"}},
        }
    ],
}

# TODO: can we surface the base op input/output schema?
_input_fit_schema = {
    "description": "Input data schema for training identity.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {}},
}

_input_predict_transform_schema = (
    {  # TODO: separate predict vs. predict_proba vs. transform
        "description": "Input data schema for transformations using identity.",
        "type": "object",
        "required": ["X", "y"],
        "additionalProperties": False,
        "properties": {"X": {}, "y": {}},
    }
)

_output_schema = {  # TODO: separate predict vs. predict_proba vs. transform
    "description": "Output data schema for transformations using identity.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.identity.html",
    "import_from": "lale.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_transform_schema,
        "output_predict": _output_schema,
        "input_predict_proba": _input_predict_transform_schema,
        "output_predict_proba": _output_schema,
        "input_transform": _input_predict_transform_schema,
        "output_transform": _output_schema,
    },
}


IdentityWrapper = lale.operators.make_operator(_IdentityWrapperImpl, _combined_schemas)

lale.docstrings.set_docstrings(IdentityWrapper)
