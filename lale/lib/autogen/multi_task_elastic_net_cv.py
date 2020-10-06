from numpy import inf, nan
from sklearn.linear_model import MultiTaskElasticNetCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class MultiTaskElasticNetCVImpl:
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
    "description": "inherited docstring for MultiTaskElasticNetCV    Multi-task L1/L2 ElasticNet with built-in cross-validation.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "l1_ratio",
                "eps",
                "n_alphas",
                "alphas",
                "fit_intercept",
                "normalize",
                "max_iter",
                "tol",
                "cv",
                "copy_X",
                "verbose",
                "n_jobs",
                "random_state",
                "selection",
            ],
            "relevantToOptimizer": [
                "eps",
                "n_alphas",
                "fit_intercept",
                "normalize",
                "max_iter",
                "tol",
                "cv",
                "copy_X",
            ],
            "additionalProperties": False,
            "properties": {
                "l1_ratio": {
                    "XXX TODO XXX": "float or array of floats",
                    "description": "The ElasticNet mixing parameter, with 0 < l1_ratio <= 1",
                    "type": "number",
                    "default": 0.5,
                },
                "eps": {
                    "type": "number",
                    "minimumForOptimizer": 0.001,
                    "maximumForOptimizer": 0.1,
                    "distribution": "loguniform",
                    "default": 0.001,
                    "description": "Length of the path",
                },
                "n_alphas": {
                    "type": "integer",
                    "minimumForOptimizer": 100,
                    "maximumForOptimizer": 101,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Number of alphas along the regularization path",
                },
                "alphas": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                            "XXX TODO XXX": "array-like, optional",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "List of alphas where to compute the models",
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
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "The maximum number of iterations",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "The tolerance for the optimization: if the updates are smaller than ``tol``, the optimization code checks the dual gap for optimality and continues until it is smaller than ``tol``.",
                },
                "cv": {
                    "XXX TODO XXX": "int, cross-validation generator or an iterable, optional",
                    "description": "Determines the cross-validation splitting strategy",
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 4,
                    "distribution": "uniform",
                    "default": 3,
                },
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If ``True``, X will be copied; else, it may be overwritten.",
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": 0,
                    "description": "Amount of verbosity.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "Number of CPUs to use during the cross validation",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator that selects a random feature to update",
                },
                "selection": {
                    "type": "string",
                    "default": "cyclic",
                    "description": "If set to 'random', a random coefficient is updated every iteration rather than looping over features sequentially by default",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: n_jobs > only if multiple values for l1_ratio are given"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit linear model with coordinate descent",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "{array-like}, shape (n_samples, n_features)",
            "description": "Training data",
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV#sklearn-linear_model-multitaskelasticnetcv",
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
set_docstrings(MultiTaskElasticNetCVImpl, _combined_schemas)
MultiTaskElasticNetCV = make_operator(MultiTaskElasticNetCVImpl, _combined_schemas)
