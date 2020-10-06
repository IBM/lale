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

import sklearn.svm

import lale.docstrings
import lale.operators


class SVRImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.svm.SVR(**self._hyperparams)

    def fit(self, X, y=None, sample_weight=None):
        self._wrapped_model.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kernel",
                "degree",
                "gamma",
                "coef0",
                "tol",
                "C",
                "epsilon",
                "shrinking",
                "cache_size",
                "verbose",
                "max_iter",
            ],
            "relevantToOptimizer": [
                "kernel",
                "degree",
                "gamma",
                "C",
                "shrinking",
                "tol",
            ],
            "properties": {
                "kernel": {
                    "anyOf": [
                        {"enum": ["precomputed"], "forOptimizer": False},
                        {"enum": ["linear", "poly", "rbf", "sigmoid"]},
                        {"laleType": "callable", "forOptimizer": False},
                    ],
                    "default": "rbf",
                    "description": "Specifies the kernel type to be used in the algorithm.",
                },
                "degree": {
                    "type": "integer",
                    "minimum": 0,
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 5,
                    "default": 3,
                    "description": "Degree of the polynomial kernel function ('poly').",
                },
                "gamma": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "minimumForOptimizer": 3.0517578125e-05,
                            "maximumForOptimizer": 8,
                            "distribution": "loguniform",
                        },
                        {"enum": ["auto", "auto_deprecated", "scale"]},
                    ],
                    "default": "auto_deprecated",  # going to change to 'scale' from sklearn 0.22.
                    "description": "Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.",
                },
                "coef0": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Independent term in kernel function.",
                },
                "tol": {
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximumForOptimizer": 0.01,
                    "default": 0.001,
                    "description": "Tolerance for stopping criteria.",
                },
                "C": {
                    "description": "Penalty parameter C of the error term.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "epsilon": {
                    "description": "Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.",
                    "type": "number",
                    "default": 0.1,
                    "minimum": 0.0,
                    "minimumForOptimizer": 0.00001,
                    "maximumForOptimizer": 10000.0,
                },
                "shrinking": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use the shrinking heuristic.",
                },
                "cache_size": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 200.0,
                    "description": "Specify the size of the kernel cache (in MB).",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable verbose output.",
                },
                "max_iter": {
                    "type": "integer",
                    "default": -1,
                    "description": "Hard limit on iterations within solver, or -1 for no limit.",
                },
            },
        },
        {
            "description": "coef0 only significant in kernel ‘poly’ and ‘sigmoid’.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"kernel": {"enum": ["poly", "sigmoid"]}},
                },
                {"type": "object", "properties": {"coef0": {"enum": [0.0]}}},
            ],
        },
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "y": {"type": "array", "items": {"type": "number"}},
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        }
    },
}

_output_predict_schema = {
    "description": "The predicted classes.",
    "type": "array",
    "items": {"type": "number"},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Support Vector Classification`_ from scikit-learn.

.. _`Support Vector Classification`: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.svc.html",
    "import_from": "sklearn.svm",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

SVR: lale.operators.IndividualOp
SVR = lale.operators.make_operator(SVRImpl, _combined_schemas)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.svm.SVR.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.svm.SVR.html
    from lale.schemas import AnyOf, Enum, Float

    SVR = SVR.customize_schema(
        gamma=AnyOf(
            types=[
                Enum(["scale", "auto"]),
                Float(
                    min=0.0,
                    exclusiveMin=True,
                    minForOptimizer=3.0517578125e-05,
                    maxForOptimizer=8,
                    distribution="loguniform",
                ),
            ],
            desc="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.",
            default="scale",
        )
    )

lale.docstrings.set_docstrings(SVRImpl, SVR._schemas)
