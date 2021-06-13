from sklearn.linear_model import ARDRegression as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "ARDRegression Hyperparameter schema.",
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
                "threshold_lambda",
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
                "threshold_lambda": {
                    "type": "number",
                    "default": 10000.0,
                    "description": "threshold for removing (pruning) weights with high precision from the computation",
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
    "description": "Fit the ARDRegression model according to the given training data",
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
            "description": "Target values (integers)",
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
    "description": """`ARDRegression`_ Bayesian ARD regression from sklearn.

.. _`ARDRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression"
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ard_regression.html",
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
ARDRegression = make_operator(Op, _combined_schemas)

set_docstrings(ARDRegression)
