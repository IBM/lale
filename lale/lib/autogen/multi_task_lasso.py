from numpy import inf, nan
from sklearn.linear_model import MultiTaskLasso as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _MultiTaskLassoImpl:
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
    "description": "inherited docstring for MultiTaskLasso    Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "alpha",
                "fit_intercept",
                "normalize",
                "copy_X",
                "max_iter",
                "tol",
                "warm_start",
                "random_state",
                "selection",
            ],
            "relevantToOptimizer": [
                "alpha",
                "fit_intercept",
                "normalize",
                "copy_X",
                "max_iter",
                "tol",
            ],
            "additionalProperties": False,
            "properties": {
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1.0,
                    "description": "Constant that multiplies the L1/L2 term",
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
                    "description": "If ``True``, X will be copied; else, it may be overwritten.",
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
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to ``True``, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution",
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
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit MultiTaskElasticNet model with coordinate descent",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "ndarray, shape (n_samples, n_features)",
            "description": "Data",
        },
        "y": {
            "laleType": "Any",
            "XXX TODO XXX": "ndarray, shape (n_samples, n_tasks)",
            "description": "Target",
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.MultiTaskLasso#sklearn-linear_model-multitasklasso",
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
MultiTaskLasso = make_operator(_MultiTaskLassoImpl, _combined_schemas)

set_docstrings(MultiTaskLasso)
