# Copyright 2019-2022 IBM Corporation
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

import numpy as np
import sklearn.linear_model

import lale.operators

from .mock_module import CustomOrigOperator as model_to_be_wrapped


class _IncreaseRowsImpl:
    def __init__(self, n_rows=5):
        self.n_rows = n_rows

    def transform(self, X, y=None):
        subset_X = X[0 : self.n_rows]
        output_X = np.concatenate((X, subset_X), axis=0)
        return output_X

    def transform_X_y(self, X, y):
        output_X = self.transform(X)
        if y is None:
            output_y = None
        else:
            subset_y = y[0 : self.n_rows]
            output_y = np.concatenate((y, subset_y), axis=0)
        return output_X, output_y


_input_transform_schema_ir = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
}

_output_transform_schema_ir = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_input_transform_X_y_schema_ir = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "anyOf": [
                {"enum": [None]},
                {"type": "array", "items": {"type": "number"}},
            ],
        },
    },
}

_output_transform_X_y_schema_ir = {
    "type": "array",
    "laleType": "tuple",
    "items": [
        {
            "description": "X",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        {
            "description": "y",
            "anyOf": [
                {"enum": [None]},
                {"type": "array", "items": {"type": "number"}},
            ],
        },
    ],
}

_hyperparam_schema_ir = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {"n_rows": {"type": "integer", "minimum": 0, "default": 5}},
        }
    ],
}

_combined_schemas_ir = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparam_schema_ir,
        "input_transform": _input_transform_schema_ir,
        "output_transform": _output_transform_schema_ir,
        "input_transform_X_y": _input_transform_X_y_schema_ir,
        "output_transform_X_y": _output_transform_X_y_schema_ir,
    },
}

IncreaseRows = lale.operators.make_operator(_IncreaseRowsImpl, _combined_schemas_ir)


class _MyLRImpl:
    _wrapped_model: sklearn.linear_model.LogisticRegression

    def __init__(self, penalty="l2", solver="liblinear", C=1.0):
        self.penalty = penalty
        self.solver = solver
        self.C = C

    def fit(self, X, y):
        result = _MyLRImpl(self.penalty, self.solver, self.C)
        result._wrapped_model = sklearn.linear_model.LogisticRegression(
            penalty=self.penalty, solver=self.solver, C=self.C
        )
        result._wrapped_model.fit(X, y)
        return result

    def predict(self, X):
        assert hasattr(self, "_wrapped_model")
        return self._wrapped_model.predict(X)


_input_fit_schema_lr = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "y": {"type": "array", "items": {"type": "number"}},
    },
}

_input_predict_schema_lr = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}

_output_predict_schema_lr = {"type": "array", "items": {"type": "number"}}

_hyperparams_ranges = {
    "type": "object",
    "additionalProperties": False,
    "required": ["solver", "penalty", "C"],
    "relevantToOptimizer": ["solver", "penalty", "C"],
    "properties": {
        "solver": {
            "description": "Algorithm for optimization problem.",
            "enum": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "default": "liblinear",
        },
        "penalty": {
            "description": "Norm used in the penalization.",
            "enum": ["l1", "l2"],
            "default": "l2",
        },
        "C": {
            "description": "Inverse regularization strength. Smaller values specify "
            "stronger regularization.",
            "type": "number",
            "distribution": "loguniform",
            "minimum": 0.0,
            "exclusiveMinimum": True,
            "default": 1.0,
            "minimumForOptimizer": 0.03125,
            "maximumForOptimizer": 32768,
        },
    },
}

_hyperparams_constraints = {
    "allOf": [
        {
            "description": "The newton-cg, sag, and lbfgs solvers support only l2 penalties.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "solver": {"not": {"enum": ["newton-cg", "sag", "lbfgs"]}}
                    },
                },
                {"type": "object", "properties": {"penalty": {"enum": ["l2"]}}},
            ],
        }
    ]
}

_hyperparams_schema_lr = {"allOf": [_hyperparams_ranges, _hyperparams_constraints]}

_combined_schemas_lr = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "input_fit": _input_fit_schema_lr,
        "input_predict": _input_predict_schema_lr,
        "output_predict": _output_predict_schema_lr,
        "hyperparams": _hyperparams_schema_lr,
    },
}

MyLR = lale.operators.make_operator(_MyLRImpl, _combined_schemas_lr)


class _CustomParamsCheckerOpImpl:
    def __init__(self, fit_params=None, predict_params=None):
        self._fit_params = fit_params
        self._predict_params = predict_params

    def fit(self, X, y=None, **kwargs):
        result = _CustomParamsCheckerOpImpl(kwargs, self._predict_params)
        return result

    def predict(self, X, **predict_params):
        self._predict_params = predict_params


_input_fit_schema_cp = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "y": {"items": {"type": "array", "items": {"type": "number"}}},
    },
}

_input_predict_schema_cp = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "number",
        }
    },
}

_output_predict_schema_cp = {
    "description": "Returns predicted values.",
    "type": "array",
    "items": {"type": "number"},
}

_hyperparam_schema_cp = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {},
        }
    ],
}

_combined_schemas_cp = {
    "description": "Combined schema for expected data and hyperparameters.",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparam_schema_cp,
        "input_fit": _input_fit_schema_cp,
        "input_predict": _input_predict_schema_cp,
        "output_predict": _output_predict_schema_cp,
    },
}

CustomParamsCheckerOp = lale.operators.make_operator(
    _CustomParamsCheckerOpImpl, _combined_schemas_cp
)

CustomOrigOperator = lale.operators.make_operator(model_to_be_wrapped, {})


class _OpThatWorksWithFilesImpl:
    def __init__(self, ngram_range):
        self.ngram_range = ngram_range

    def fit(self, X, y=None, sample_weight=None):
        import pandas as pd

        assert (
            sample_weight is not None
        ), "This is to test that Hyperopt passes fit_params correctly."
        from lale.lib.sklearn import LogisticRegression, TfidfVectorizer

        self.pipeline = (
            TfidfVectorizer(input="content", ngram_range=self.ngram_range)
            >> LogisticRegression()
        )
        self._wrapped_model = self.pipeline.fit(pd.read_csv(X, header=None), y)
        return self

    def predict(self, X):
        assert hasattr(self, "_wrapped_model")
        import pandas as pd

        return self._wrapped_model.predict(pd.read_csv(X, header=None))


_hyperparams_ranges_OpThatWorksWithFilesImpl = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ngram_range"],
    "relevantToOptimizer": ["ngram_range"],
    "properties": {
        "ngram_range": {
            "default": (1, 1),
            "anyOf": [
                {
                    "type": "array",
                    "laleType": "tuple",
                    "minItemsForOptimizer": 2,
                    "maxItemsForOptimizer": 2,
                    "items": {
                        "type": "integer",
                        "minimumForOptimizer": 1,
                        "maximumForOptimizer": 3,
                    },
                    "forOptimizer": False,
                },
                {"enum": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]},
            ],
        },
    },
}

_hyperparams_schema_OpThatWorksWithFilesImpl = {
    "allOf": [_hyperparams_ranges_OpThatWorksWithFilesImpl]
}

_combined_schemas_OpThatWorksWithFilesImpl = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "input_fit": {},
        "input_predict": {},
        "output_predict": {},
        "hyperparams": _hyperparams_schema_OpThatWorksWithFilesImpl,
    },
}

OpThatWorksWithFiles = lale.operators.make_operator(
    _OpThatWorksWithFilesImpl, _combined_schemas_OpThatWorksWithFilesImpl
)
