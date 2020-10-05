from numpy import inf, nan
from sklearn.linear_model import Lars as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class LarsImpl:
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
    "description": "inherited docstring for Lars    Least Angle Regression model a.k.a. LAR",
    "allOf": [
        {
            "type": "object",
            "required": [
                "fit_intercept",
                "verbose",
                "normalize",
                "precompute",
                "n_nonzero_coefs",
                "eps",
                "copy_X",
                "fit_path",
                "jitter",
                "random_state",
            ],
            "relevantToOptimizer": [
                "fit_intercept",
                "normalize",
                "precompute",
                "n_nonzero_coefs",
                "eps",
                "copy_X",
            ],
            "additionalProperties": False,
            "properties": {
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model",
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": False,
                    "description": "Sets the verbosity amount",
                },
                "normalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False",
                },
                "precompute": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                            "XXX TODO XXX": "True | False | 'auto' | array-like",
                            "forOptimizer": False,
                        },
                        {"enum": ["auto"]},
                    ],
                    "default": "auto",
                    "description": "Whether to use a precomputed Gram matrix to speed up calculations",
                },
                "n_nonzero_coefs": {
                    "type": "integer",
                    "minimumForOptimizer": 500,
                    "maximumForOptimizer": 501,
                    "distribution": "uniform",
                    "default": 500,
                    "description": "Target number of non-zero coefficients",
                },
                "eps": {
                    "type": "number",
                    "minimumForOptimizer": 0.001,
                    "maximumForOptimizer": 0.1,
                    "distribution": "loguniform",
                    "default": 2.220446049250313e-16,
                    "description": "The machine-precision regularization in the computation of the Cholesky diagonal factors",
                },
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If ``True``, X will be copied; else, it may be overwritten.",
                },
                "fit_path": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True the full path is stored in the ``coef_path_`` attribute",
                },
                "jitter": {
                    "anyOf": [{"type": "number"}, {"enum": [None]},],
                    "default": None,
                    "description": "Upper bound on a uniform noise parameter to be added to the y values, to satisfy the model’s assumption of one-at-a-time computations",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator to use when shuffling the data",
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
            "description": "Target values.",
        },
        "Xy": {
            "laleType": "Any",
            "XXX TODO XXX": "array-like, shape (n_samples,) or (n_samples, n_targets),                 optional",
            "description": "Xy = np.dot(X.T, y) that can be precomputed",
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.Lars#sklearn-linear_model-lars",
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
set_docstrings(LarsImpl, _combined_schemas)
Lars = make_operator(LarsImpl, _combined_schemas)
