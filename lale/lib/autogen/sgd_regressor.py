from numpy import inf, nan
from sklearn.linear_model import SGDRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _SGDRegressorImpl:
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
    "description": "inherited docstring for SGDRegressor    Linear model fitted by minimizing a regularized empirical loss with SGD",
    "allOf": [
        {
            "type": "object",
            "required": [
                "loss",
                "penalty",
                "alpha",
                "l1_ratio",
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "verbose",
                "epsilon",
                "random_state",
                "learning_rate",
                "eta0",
                "power_t",
                "early_stopping",
                "validation_fraction",
                "n_iter_no_change",
                "warm_start",
                "average",
            ],
            "relevantToOptimizer": [
                "loss",
                "penalty",
                "alpha",
                "fit_intercept",
                "max_iter",
                "tol",
                "shuffle",
                "epsilon",
                "learning_rate",
                "eta0",
            ],
            "additionalProperties": False,
            "properties": {
                "loss": {
                    "enum": [
                        "epsilon_insensitive",
                        "huber",
                        "squared_epsilon_insensitive",
                        "squared_loss",
                    ],
                    "default": "squared_loss",
                    "description": "The loss function to be used",
                },
                "penalty": {
                    "XXX TODO XXX": "str, 'none', 'l2', 'l1', or 'elasticnet'",
                    "description": "The penalty (aka regularization term) to be used",
                    "enum": ["elasticnet", "l1", "l2", "none"],
                    "default": "l2",
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Constant that multiplies the regularization term",
                },
                "l1_ratio": {
                    "type": "number",
                    "default": 0.15,
                    "description": "The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1",
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
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not the training data should be shuffled after each epoch",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "The verbosity level.",
                },
                "epsilon": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 0.1,
                    "description": "Epsilon in the epsilon-insensitive loss functions; only if `loss` is 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'",
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
                "learning_rate": {
                    "enum": ["adaptive", "constant", "optimal", "invscaling"],
                    "default": "invscaling",
                    "description": "The learning rate schedule:  'constant':     eta = eta0 'optimal':     eta = 1.0 / (alpha * (t + t0))     where t0 is chosen by a heuristic proposed by Leon Bottou",
                },
                "eta0": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.01,
                    "description": "The initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules",
                },
                "power_t": {
                    "type": "number",
                    "default": 0.25,
                    "description": "The exponent for inverse scaling learning rate [default 0.5].",
                },
                "early_stopping": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use early stopping to terminate training when validation score is not improving",
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
            "description": "epsilon, only if loss is 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'",
            "anyOf": [
                {"type": "object", "properties": {"epsilon": {"enum": [0.1]}}},
                {
                    "type": "object",
                    "properties": {
                        "loss": {
                            "enum": [
                                "huber",
                                "epsilon_insensitive",
                                "squared_epsilon_insensitive",
                            ]
                        }
                    },
                },
            ],
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
    "description": "Fit linear model with Stochastic Gradient Descent.",
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
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Weights applied to individual samples (1",
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.SGDRegressor#sklearn-linear_model-sgdregressor",
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
SGDRegressor = make_operator(_SGDRegressorImpl, _combined_schemas)

set_docstrings(SGDRegressor)
