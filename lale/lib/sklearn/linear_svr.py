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

import sklearn.svm

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Hyperparam schema for LinearSVR (Linear Support Vector Regression).",
    "allOf": [
        {
            "type": "object",
            "required": [
                "epsilon",
                "tol",
                "C",
                "loss",
                "fit_intercept",
                "intercept_scaling",
                "dual",
                "verbose",
                "random_state",
                "max_iter",
            ],
            "relevantToOptimizer": [
                "epsilon",
                "tol",
                "loss",
                "fit_intercept",
                "dual",
                "max_iter",
            ],
            "additionalProperties": False,
            "properties": {
                "epsilon": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 0.0,
                    "description": """Epsilon parameter in the epsilon-insensitive loss function.
Note that the value of this parameter depends on the scale of the target variable y. If unsure, set epsilon=0.""",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance for stopping criteria.",
                },
                "C": {
                    "type": "number",
                    "default": 1.0,
                    "description": """Regularization parameter.
The strength of the regularization is inversely proportional to C. Must be strictly positive.""",
                },
                "loss": {
                    "enum": [
                        "hinge",
                        "squared_epsilon_insensitive",
                        "squared_hinge",
                        "epsilon_insensitive",
                    ],
                    "default": "epsilon_insensitive",
                    "description": """Specifies the loss function.
The epsilon-insensitive loss (standard SVR) is the L1 loss, while the squared epsilon-insensitive loss (‘squared_epsilon_insensitive’) is the L2 loss.""",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": """Whether to calculate the intercept for this model.
If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).""",
                },
                "intercept_scaling": {
                    "type": "number",
                    "default": 1.0,
                    "description": """When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling],
i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector.
The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features.
To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.""",
                },
                "dual": {
                    "type": "boolean",
                    "default": True,
                    "description": """Select the algorithm to either solve the dual or primal optimization problem.
Prefer dual=False when n_samples > n_features.""",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": """Enable verbose output.
Note that this setting takes advantage of a per-process runtime setting in liblinear that, if enabled, may not work properly in a multithreaded context.""",
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator.",
                    "anyOf": [
                        {"laleType": "numpy.random.RandomState"},
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "The maximum number of iterations to be run.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model according to the given training data.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vector, where n_samples in the number of samples and n_features is the number of features.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target vector relative to X",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Array of weights that are assigned to individual samples",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model",
    "type": "object",
    "required": ["X"],
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
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns predicted values.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`LinearSVR`_ from scikit-learn.

.. _`LinearSVR`: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.linear_svr.html",
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

LinearSVR = lale.operators.make_operator(sklearn.svm.LinearSVR, _combined_schemas)

lale.docstrings.set_docstrings(LinearSVR)
