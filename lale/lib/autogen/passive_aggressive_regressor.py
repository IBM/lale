from numpy import inf, nan
from sklearn.linear_model import PassiveAggressiveRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _PassiveAggressiveRegressorImpl:
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
    "description": "inherited docstring for PassiveAggressiveRegressor    Passive Aggressive Regressor",
    "allOf": [
        {
            "type": "object",
            "required": [
                "C",
                "fit_intercept",
                "max_iter",
                "tol",
                "early_stopping",
                "validation_fraction",
                "n_iter_no_change",
                "shuffle",
                "verbose",
                "loss",
                "epsilon",
                "random_state",
                "warm_start",
                "average",
            ],
            "relevantToOptimizer": [
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "loss",
                "epsilon",
            ],
            "additionalProperties": False,
            "properties": {
                "C": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Maximum step size (regularization)",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the intercept should be estimated or not",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 10,
                            "maximumForOptimizer": 1000,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The maximum number of passes over the training data (aka epochs)",
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
                    "description": "The stopping criterion",
                },
                "early_stopping": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use early stopping to terminate training when validation",
                },
                "validation_fraction": {
                    "type": "number",
                    "default": 0.1,
                    "description": "The proportion of training data to set aside as validation set for early stopping",
                },
                "n_iter_no_change": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of iterations with no improvement to wait before early stopping",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not the training data should be shuffled after each epoch.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "The verbosity level",
                },
                "loss": {
                    "enum": [
                        "huber",
                        "squared_epsilon_insensitive",
                        "squared_loss",
                        "epsilon_insensitive",
                    ],
                    "default": "epsilon_insensitive",
                    "description": "The loss function to be used: epsilon_insensitive: equivalent to PA-I in the reference paper",
                },
                "epsilon": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 0.1,
                    "description": "If the difference between the current prediction and the correct label is below this threshold, the model is not updated.",
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
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution",
                },
                "average": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": False,
                    "description": "When set to True, computes the averaged SGD weights and stores the result in the ``coef_`` attribute",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: max_iter > only impacts the behavior in the fit method, and not the partial_fit"
        },
        {
            "description": "validation_fraction, only used if early_stopping is true",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"validation_fraction": {"enum": [0.1]}},
                },
                {"type": "object", "properties": {"early_stopping": {"enum": [True]}}},
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit linear model with Passive Aggressive algorithm.",
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
        "coef_init": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The initial coefficients to warm-start the optimization.",
        },
        "intercept_init": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The initial intercept to warm-start the optimization.",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predicted target values per element in X.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor#sklearn-linear_model-passiveaggressiveregressor",
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
PassiveAggressiveRegressor = make_operator(
    _PassiveAggressiveRegressorImpl, _combined_schemas
)

set_docstrings(PassiveAggressiveRegressor)
