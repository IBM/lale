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

import sklearn
import sklearn.linear_model

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Linear least squares with l2 regularization.",
    "allOf": [
        {
            "type": "object",
            "required": ["alpha", "fit_intercept", "solver"],
            "relevantToOptimizer": [
                "alpha",
                "fit_intercept",
                "normalize",
                "copy_X",
                "max_iter",
                "tol",
                "solver",
            ],
            "additionalProperties": False,
            "properties": {
                "alpha": {
                    "description": "Regularization strength; larger values specify stronger regularization.",
                    "anyOf": [
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "minimumForOptimizer": 1e-10,
                            "maximumForOptimizer": 1.0,
                            "default": 1.0,
                            "distribution": "loguniform",
                        },
                        {
                            "type": "array",
                            "description": "Penalties specific to the targets.",
                            "items": {
                                "type": "number",
                                "minimum": 0.0,
                                "exclusiveMinimum": True,
                            },
                            "forOptimizer": False,
                        },
                    ],
                    "default": 1.0,
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model.",
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False.",
                },
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, X will be copied; else, it may be overwritten.",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "minimumForOptimizer": 10,
                            "maximumForOptimizer": 1000,
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Maximum number of iterations for conjugate gradient solver.",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.001,
                    "description": "Precision of the solution.",
                },
                "solver": {
                    "enum": [
                        "auto",
                        "svd",
                        "cholesky",
                        "lsqr",
                        "sparse_cg",
                        "sag",
                        "saga",
                    ],
                    "default": "auto",
                    "description": "Solver to use in the computational routines.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator to use when shuffling",
                },
            },
        },
        {
            "description": "solver {svd, lsqr, cholesky, saga} does not support fitting the intercept on sparse data. Please set the solver to 'auto' or 'sparse_cg', 'sag', or set `fit_intercept=False. ",
            "anyOf": [
                {"type": "object", "laleNot": "X/isSparse"},
                {"type": "object", "properties": {"fit_intercept": {"enum": [False]}}},
                {
                    "type": "object",
                    "properties": {"solver": {"enum": ["auto", "sparse_cg", "sag"]}},
                },
            ],
        },
        {
            "description": "SVD solver does not support sparse inputs currently.",
            "anyOf": [
                {"type": "object", "laleNot": "X/isSparse"},
                {
                    "type": "object",
                    "properties": {"solver": {"not": {"enum": ["svd"]}}},
                },
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Fit Ridge regression model",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Training data",
        },
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
            ],
            "description": "Target values",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "number"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "description": "Individual weights for each sample",
        },
    },
}
_input_predict_schema = {
    "description": "Predict using the linear model",
    "type": "object",
    "properties": {
        "X": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
            ],
            "description": "Samples.",
        },
    },
}
_output_predict_schema = {
    "description": "Returns predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {  # There was a case where Ridge returned 2-d predictions for a single target.
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Ridge`_ regression estimator from scikit-learn.

.. _`Ridge`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ridge.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


Ridge = lale.operators.make_operator(sklearn.linear_model.Ridge, _combined_schemas)

if sklearn.__version__ >= "1.0":
    # old: https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.Ridge.html
    # new: https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.Ridge.html

    Ridge = Ridge.customize_schema(
        relevantToOptimizer=[
            "alpha",
            "fit_intercept",
            "copy_X",
            "max_iter",
            "tol",
            "solver",
        ],
        normalize={
            "type": "boolean",
            "description": """This parameter is ignored when fit_intercept is set to False.
If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
If you wish to standardize, please use StandardScaler before calling fit on an estimator with normalize=False.""",
            "default": False,
            "forOptimizer": False,
        },
        positive={
            "type": "boolean",
            "description": """When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.""",
            "default": False,
            "forOptimizer": False,
        },
        solver={
            "enum": [
                "auto",
                "svd",
                "cholesky",
                "lsqr",
                "sparse_cg",
                "sag",
                "saga",
                "lbfgs",
            ],
            "default": "auto",
            "description": """Solver to use in the computational routines:
- 'auto' chooses the solver automatically based on the type of data.
- 'svd' uses a Singular Value Decomposition of X to compute the Ridge
    coefficients. More stable for singular matrices than 'cholesky'.
- 'cholesky' uses the standard scipy.linalg.solve function to
    obtain a closed-form solution.
- 'sparse_cg' uses the conjugate gradient solver as found in
    scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
    more appropriate than 'cholesky' for large-scale data
    (possibility to set `tol` and `max_iter`).
- 'lsqr' uses the dedicated regularized least-squares routine
    scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
    procedure.
- 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
    its improved, unbiased version named SAGA. Both methods also use an
    iterative procedure, and are often faster than other solvers when
    both n_samples and n_features are large. Note that 'sag' and
    'saga' fast convergence is only guaranteed on features with
    approximately the same scale. You can preprocess the data with a
    scaler from sklearn.preprocessing.
- 'lbfgs' uses L-BFGS-B algorithm implemented in
    `scipy.optimize.minimize`. It can be used only when `positive`
    is True.
All last six solvers support both dense and sparse data. However, only
'sag', 'sparse_cg', and 'lbfgs' support sparse input when `fit_intercept`
is True.""",
            "default": "auto",
            "forOptimizer": True,
        },
        set_as_available=True,
    )
    Ridge = Ridge.customize_schema(
        constraint={
            "description": "Only ‘lbfgs’ solver is supported when positive is True. `auto` works too when tested.",
            "anyOf": [
                {"type": "object", "properties": {"positive": {"enum": [False]}}},
                {
                    "type": "object",
                    "properties": {
                        "solver": {"enum": ["lbfgs", "auto"]},
                    },
                },
            ],
        },
        set_as_available=True,
    )

    Ridge = Ridge.customize_schema(
        constraint={
            "description": "`lbfgs` solver can be used only when positive=True.",
            "anyOf": [
                {"type": "object", "properties": {"positive": {"enum": [True]}}},
                {
                    "type": "object",
                    "properties": {
                        "solver": {"not": {"enum": ["lbfgs"]}},
                    },
                },
            ],
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(Ridge)
