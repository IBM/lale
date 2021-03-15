from numpy import inf, nan
from sklearn.linear_model import OrthogonalMatchingPursuit as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _OrthogonalMatchingPursuitImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for OrthogonalMatchingPursuit    Orthogonal Matching Pursuit model (OMP)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_nonzero_coefs",
                "tol",
                "fit_intercept",
                "normalize",
                "precompute",
            ],
            "relevantToOptimizer": [
                "n_nonzero_coefs",
                "tol",
                "fit_intercept",
                "normalize",
                "precompute",
            ],
            "additionalProperties": False,
            "properties": {
                "n_nonzero_coefs": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 500,
                            "maximumForOptimizer": 501,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Desired number of non-zero entries in the solution",
                },
                "tol": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 1e-08,
                            "maximumForOptimizer": 0.01,
                            "distribution": "loguniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Maximum norm of the residual",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "whether to calculate the intercept for this model",
                },
                "normalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False",
                },
                "precompute": {
                    "enum": [True, False, "auto"],
                    "default": "auto",
                    "description": "Whether to use a precomputed Gram and Xy matrix to speed up calculations",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model using X, y as training data.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data.",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Target values",
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
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array_like or sparse matrix, shape (n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Samples.",
        }
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
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit#sklearn-linear_model-orthogonalmatchingpursuit",
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
OrthogonalMatchingPursuit = make_operator(
    _OrthogonalMatchingPursuitImpl, _combined_schemas
)

set_docstrings(OrthogonalMatchingPursuit)
