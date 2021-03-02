from numpy import inf, nan
from sklearn.neural_network import MLPClassifier as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _MLPClassifierImpl:
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

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for MLPClassifier    Multi-layer Perceptron classifier.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "hidden_layer_sizes",
                "activation",
                "solver",
                "alpha",
                "batch_size",
                "learning_rate",
                "learning_rate_init",
                "power_t",
                "max_iter",
                "shuffle",
                "random_state",
                "tol",
                "verbose",
                "warm_start",
                "momentum",
                "nesterovs_momentum",
                "early_stopping",
                "validation_fraction",
                "beta_1",
                "beta_2",
                "epsilon",
                "n_iter_no_change",
            ],
            "relevantToOptimizer": [
                "activation",
                "solver",
                "alpha",
                "batch_size",
                "learning_rate",
                "max_iter",
                "shuffle",
                "tol",
                "nesterovs_momentum",
                "epsilon",
            ],
            "additionalProperties": False,
            "properties": {
                "hidden_layer_sizes": {
                    "XXX TODO XXX": "tuple, length = n_layers - 2, default (100,)",
                    "description": "The ith element represents the number of neurons in the ith hidden layer.",
                    "type": "array",
                    "laleType": "tuple",
                    "default": (100,),
                },
                "activation": {
                    "enum": ["identity", "logistic", "tanh", "relu"],
                    "default": "relu",
                    "description": "Activation function for the hidden layer",
                },
                "solver": {
                    "enum": ["lbfgs", "sgd", "adam"],
                    "default": "adam",
                    "description": "The solver for weight optimization",
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "L2 penalty (regularization term) parameter.",
                },
                "batch_size": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 128,
                            "distribution": "uniform",
                        },
                        {"enum": ["auto"]},
                    ],
                    "default": "auto",
                    "description": "Size of minibatches for stochastic optimizers",
                },
                "learning_rate": {
                    "enum": ["constant", "invscaling", "adaptive"],
                    "default": "constant",
                    "description": "Learning rate schedule for weight updates",
                },
                "learning_rate_init": {
                    "type": "number",
                    "default": 0.001,
                    "description": "The initial learning rate used",
                },
                "power_t": {
                    "type": "number",
                    "default": 0.5,
                    "description": "The exponent for inverse scaling learning rate",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 200,
                    "description": "Maximum number of iterations",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to shuffle samples in each iteration",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance for the optimization",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to print progress messages to stdout.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution",
                },
                "momentum": {
                    "type": "number",
                    "default": 0.9,
                    "description": "Momentum for gradient descent update",
                },
                "nesterovs_momentum": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use Nesterov's momentum",
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
                "beta_1": {
                    "type": "number",
                    "default": 0.9,
                    "description": "Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1)",
                },
                "beta_2": {
                    "type": "number",
                    "default": 0.999,
                    "description": "Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1)",
                },
                "epsilon": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 1e-08,
                    "description": "Value for numerical stability in adam",
                },
                "n_iter_no_change": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of epochs to not meet ``tol`` improvement",
                },
            },
        },
        {
            "description": "learning_rate, only used when solver='sgd'",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"learning_rate": {"enum": ["constant"]}},
                },
                {"type": "object", "properties": {"solver": {"enum": ["sgd"]}}},
            ],
        },
        {
            "description": "learning_rate_init, only used when solver='sgd' or 'adam'",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"learning_rate_init": {"enum": [0.001]}},
                },
                {"type": "object", "properties": {"solver": {"enum": ["sgd", "adam"]}}},
            ],
        },
        {
            "description": "power_t, only used when solver='sgd'",
            "anyOf": [
                {"type": "object", "properties": {"power_t": {"enum": [0.5]}}},
                {"type": "object", "properties": {"solver": {"enum": ["sgd"]}}},
            ],
        },
        {
            "description": "shuffle, only used when solver='sgd' or 'adam'",
            "anyOf": [
                {"type": "object", "properties": {"shuffle": {"enum": [True]}}},
                {"type": "object", "properties": {"solver": {"enum": ["sgd", "adam"]}}},
            ],
        },
        {
            "description": "momentum, only used when solver='sgd'",
            "anyOf": [
                {"type": "object", "properties": {"momentum": {"enum": [0.9]}}},
                {"type": "object", "properties": {"solver": {"enum": ["sgd"]}}},
            ],
        },
        {
            "XXX TODO XXX": "Parameter: nesterovs_momentum > only used when solver='sgd' and momentum > 0"
        },
        {
            "description": "early_stopping, only effective when solver='sgd' or 'adam'",
            "anyOf": [
                {"type": "object", "properties": {"early_stopping": {"enum": [False]}}},
                {"type": "object", "properties": {"solver": {"enum": ["sgd", "adam"]}}},
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
        {
            "description": "beta_1, only used when solver='adam'",
            "anyOf": [
                {"type": "object", "properties": {"beta_1": {"enum": [0.9]}}},
                {"type": "object", "properties": {"solver": {"enum": ["adam"]}}},
            ],
        },
        {
            "description": "beta_2, only used when solver='adam'",
            "anyOf": [
                {"type": "object", "properties": {"beta_2": {"enum": [0.999]}}},
                {"type": "object", "properties": {"solver": {"enum": ["adam"]}}},
            ],
        },
        {
            "description": "epsilon, only used when solver='adam'",
            "anyOf": [
                {"type": "object", "properties": {"epsilon": {"enum": [1e-08]}}},
                {"type": "object", "properties": {"solver": {"enum": ["adam"]}}},
            ],
        },
        {
            "description": "n_iter_no_change, only effective when solver='sgd' or 'adam' ",
            "anyOf": [
                {"type": "object", "properties": {"n_iter_no_change": {"enum": [10]}}},
                {"type": "object", "properties": {"solver": {"enum": ["sgd", "adam"]}}},
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model to data matrix X and target(s) y.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix, shape (n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The input data.",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The target values (class labels in classification, real numbers in regression).",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the multi-layer perceptron classifier",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The input data.",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Probability estimates.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The input data.",
        }
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted probability of the sample for each class in the model, where classes are ordered as they are in `self.classes_`.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.neural_network.MLPClassifier#sklearn-neural_network-mlpclassifier",
    "import_from": "sklearn.neural_network",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}
MLPClassifier = make_operator(_MLPClassifierImpl, _combined_schemas)

set_docstrings(MLPClassifier)
