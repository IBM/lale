from numpy import inf, nan
from sklearn.neural_network import BernoulliRBM as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class BernoulliRBMImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for BernoulliRBM    Bernoulli Restricted Boltzmann Machine (RBM).",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "learning_rate",
                "batch_size",
                "n_iter",
                "verbose",
                "random_state",
            ],
            "relevantToOptimizer": ["n_components", "batch_size", "n_iter"],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 256,
                    "distribution": "uniform",
                    "default": 256,
                    "description": "Number of binary hidden units.",
                },
                "learning_rate": {
                    "type": "number",
                    "default": 0.1,
                    "description": "The learning rate for weight updates",
                },
                "batch_size": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 128,
                    "distribution": "uniform",
                    "default": 10,
                    "description": "Number of examples per minibatch.",
                },
                "n_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 10,
                    "description": "Number of iterations/sweeps over the training dataset to perform during training.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "The verbosity level",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                    ],
                    "default": 33,
                    "description": "A random number generator instance to define the state of the random permutations generator",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model to the data X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the hidden layer activation probabilities, P(h=1|v=X).",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data to be transformed.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Latent representations of the data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.neural_network.BernoulliRBM#sklearn-neural_network-bernoullirbm",
    "import_from": "sklearn.neural_network",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
set_docstrings(BernoulliRBMImpl, _combined_schemas)
BernoulliRBM = make_operator(BernoulliRBMImpl, _combined_schemas)
