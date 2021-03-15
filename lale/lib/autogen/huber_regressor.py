from numpy import inf, nan
from sklearn.linear_model import HuberRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _HuberRegressorImpl:
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
    "description": "inherited docstring for HuberRegressor    Linear regression model that is robust to outliers.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "epsilon",
                "max_iter",
                "alpha",
                "warm_start",
                "fit_intercept",
                "tol",
            ],
            "relevantToOptimizer": [
                "epsilon",
                "max_iter",
                "alpha",
                "fit_intercept",
                "tol",
            ],
            "additionalProperties": False,
            "properties": {
                "epsilon": {
                    "XXX TODO XXX": "float, greater than 1.0, default 1.35",
                    "description": "The parameter epsilon controls the number of samples that should be classified as outliers",
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 1.35,
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Maximum number of iterations that scipy.optimize.fmin_l_bfgs_b should run for.",
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Regularization parameter.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "This is useful if the stored attributes of a previously used model has to be reused",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not to fit the intercept",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 1e-05,
                    "description": "The iteration will stop when ``max{|proj g_i | i = 1, ",
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
            "description": "Target vector relative to X.",
        },
        "sample_weight": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Weight given to each sample.",
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.HuberRegressor#sklearn-linear_model-huberregressor",
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
HuberRegressor = make_operator(_HuberRegressorImpl, _combined_schemas)

set_docstrings(HuberRegressor)
