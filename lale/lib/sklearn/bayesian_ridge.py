import typing

import sklearn
from sklearn.linear_model import BayesianRidge as Op

import lale.operators
from lale.docstrings import set_docstrings
from lale.operators import make_operator
from lale.schemas import AnyOf, Float, Null

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Bayesian ridge regression",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_iter",
                "tol",
                "alpha_1",
                "alpha_2",
                "lambda_1",
                "lambda_2",
                "compute_score",
                "fit_intercept",
                "normalize",
                "copy_X",
                "verbose",
            ],
            "relevantToOptimizer": [
                "n_iter",
                "tol",
                "compute_score",
                "fit_intercept",
                "normalize",
            ],
            "additionalProperties": False,
            "properties": {
                "n_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 300,
                    "description": "Maximum number of iterations",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.001,
                    "description": "Stop the algorithm if w has converged",
                },
                "alpha_1": {
                    "type": "number",
                    "default": 1e-06,
                    "description": "Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter",
                },
                "alpha_2": {
                    "type": "number",
                    "default": 1e-06,
                    "description": "Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter",
                },
                "lambda_1": {
                    "type": "number",
                    "default": 1e-06,
                    "description": "Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter",
                },
                "lambda_2": {
                    "type": "number",
                    "default": 1e-06,
                    "description": "Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter",
                },
                "compute_score": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, compute the objective function at each step of the model",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "whether to calculate the intercept for this model",
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False",
                },
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, X will be copied; else, it may be overwritten.",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Verbose mode when fitting the model.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values",
        },
        "sample_weight": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Individual weights for each sample  ",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Samples.",
        },
        "return_std": {
            "anyOf": [{"type": "boolean"}, {"enum": [None]}],
            "default": None,
            "description": "Whether to return the standard deviation of posterior prediction.",
        },
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model.",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Bayesian ridge regression`_ from sklearn

.. _`Bayesian ridge regression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.bayesian_ridge.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
BayesianRidge = make_operator(Op, _combined_schemas)

if sklearn.__version__ >= "0.22":
    BayesianRidge = typing.cast(
        lale.operators.PlannedIndividualOp,
        BayesianRidge.customize_schema(
            alpha_init=AnyOf(
                types=[Float(), Null()],
                desc="Initial value for alpha (precision of the noise). If not set, alpha_init is 1/Var(y).",
                default=None,
            ),
            lambda_init=AnyOf(
                types=[Float(), Null()],
                desc="Initial value for lambda (precision of the weights). If not set, lambda_init is 1.",
                default=None,
            ),
        ),
    )


set_docstrings(BayesianRidge)