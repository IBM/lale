from numpy import inf, nan
from sklearn.linear_model import LassoCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _LassoCVImpl:
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
    "description": "inherited docstring for LassoCV    Lasso linear model with iterative fitting along a regularization path.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "eps",
                "n_alphas",
                "alphas",
                "fit_intercept",
                "normalize",
                "precompute",
                "max_iter",
                "tol",
                "copy_X",
                "cv",
                "verbose",
                "n_jobs",
                "positive",
                "random_state",
                "selection",
            ],
            "relevantToOptimizer": [
                "eps",
                "n_alphas",
                "fit_intercept",
                "normalize",
                "precompute",
                "max_iter",
                "tol",
                "copy_X",
                "cv",
                "positive",
                "selection",
            ],
            "additionalProperties": False,
            "properties": {
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
                            "XXX TODO XXX": "numpy array, optional",
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
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If ``True``, X will be copied; else, it may be overwritten.",
                },
                "cv": {
                    "description": """Cross-validation as integer or as object that has a split function.
                        The fit method performs cross validation on the input dataset for per
                        trial, and uses the mean cross validation performance for optimization.
                        This behavior is also impacted by handle_cv_failure flag.
                        If integer: number of folds in sklearn.model_selection.StratifiedKFold.
                        If object with split function: generator yielding (train, test) splits
                        as arrays of indices. Can use any of the iterators from
                        https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators.""",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "default": 5,
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 4,
                            "distribution": "uniform",
                        },
                        {"laleType": "Any", "forOptimizer": False},
                    ],
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": False,
                    "description": "Amount of verbosity.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "Number of CPUs to use during the cross validation",
                },
                "positive": {
                    "type": "boolean",
                    "default": False,
                    "description": "If positive, restrict regression coefficients to be positive",
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
                    "enum": ["random", "cyclic"],
                    "default": "cyclic",
                    "description": "If set to 'random', a random coefficient is updated every iteration rather than looping over features sequentially by default",
                },
            },
        }
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.LassoCV#sklearn-linear_model-lassocv",
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
LassoCV = make_operator(_LassoCVImpl, _combined_schemas)

set_docstrings(LassoCV)
