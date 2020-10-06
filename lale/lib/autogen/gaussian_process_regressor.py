from numpy import inf, nan
from sklearn.gaussian_process import GaussianProcessRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class GaussianProcessRegressorImpl:
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
    "description": "inherited docstring for GaussianProcessRegressor    Gaussian process regression (GPR).",
    "allOf": [
        {
            "type": "object",
            "required": [
                "kernel",
                "alpha",
                "optimizer",
                "n_restarts_optimizer",
                "normalize_y",
                "copy_X_train",
                "random_state",
            ],
            "relevantToOptimizer": [
                "alpha",
                "optimizer",
                "n_restarts_optimizer",
                "normalize_y",
            ],
            "additionalProperties": False,
            "properties": {
                "kernel": {
                    "XXX TODO XXX": "kernel object",
                    "description": "The kernel specifying the covariance function of the GP",
                    "enum": [None],
                    "default": None,
                },
                "alpha": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 1e-10,
                            "maximumForOptimizer": 1.0,
                            "distribution": "loguniform",
                        },
                        {
                            "type": "array",
                            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                            "XXX TODO XXX": "float or array-like, optional (default: 1e-10)",
                            "forOptimizer": False,
                        },
                    ],
                    "default": 1e-10,
                    "description": "Value added to the diagonal of the kernel matrix during fitting",
                },
                "optimizer": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": ["fmin_l_bfgs_b"]},
                    ],
                    "default": "fmin_l_bfgs_b",
                    "description": "Can either be one of the internally supported optimizers for optimizing the kernel's parameters, specified by a string, or an externally defined optimizer passed as a callable",
                },
                "n_restarts_optimizer": {
                    "type": "integer",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                    "distribution": "uniform",
                    "default": 0,
                    "description": "The number of restarts of the optimizer for finding the kernel's parameters which maximize the log-marginal likelihood",
                },
                "normalize_y": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the target values y are normalized, i.e., the mean of the observed target values become zero",
                },
                "copy_X_train": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, a persistent copy of the training data is stored in the object",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The generator used to initialize the centers",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit Gaussian process regression model.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data",
        },
        "y": {
            "laleType": "Any",
            "XXX TODO XXX": "array-like, shape = (n_samples, [n_output_dims])",
            "description": "Target values",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the Gaussian process regression model",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Query points where the GP is evaluated",
        },
        "return_std": {
            "type": "boolean",
            "default": False,
            "description": "If True, the standard-deviation of the predictive distribution at the query points is returned along with the mean.",
        },
        "return_cov": {
            "type": "boolean",
            "default": False,
            "description": "If True, the covariance of the joint predictive distribution at the query points is returned along with the mean",
        },
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the Gaussian process regression model",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor#sklearn-gaussian_process-gaussianprocessregressor",
    "import_from": "sklearn.gaussian_process",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
set_docstrings(GaussianProcessRegressorImpl, _combined_schemas)
GaussianProcessRegressor = make_operator(
    GaussianProcessRegressorImpl, _combined_schemas
)
