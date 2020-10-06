# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sklearn.neural_network

import lale.docstrings
import lale.operators


class MLPClassifierImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.neural_network.MLPClassifier(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def partial_fit(self, X, y=None, classes=None):
        if not hasattr(self, "_wrapped_model"):
            self._wrapped_model = sklearn.neural_network.MLPClassifier(
                **self._hyperparams
            )
        self._wrapped_model.partial_fit(X, y, classes=classes)
        return self


_hyperparams_schema = {
    "description": "Hyperparameter schema for the MLPClassifier model from scikit-learn.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
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
                "hidden_layer_sizes",
                "activation",
                "solver",
                "alpha",
                "batch_size",
                "learning_rate",
                "tol",
                "momentum",
                "nesterovs_momentum",
                "early_stopping",
                "validation_fraction",
                "beta_1",
                "beta_2",
                "epsilon",
            ],
            "properties": {
                "hidden_layer_sizes": {
                    "description": "The ith element represents the number of neurons in "
                    "the ith hidden layer.",
                    "type": "array",
                    "laleType": "tuple",
                    "minItemsForOptimizer": 1,
                    "maxItemsForOptimizer": 20,
                    "items": {
                        "type": "integer",
                        "minimumForOptimizer": 1,
                        "maximumForOptimizer": 500,
                    },
                    "default": [100],
                },
                "activation": {
                    "description": "Activation function for the hidden layer.",
                    "enum": ["identity", "logistic", "tanh", "relu"],
                    "default": "relu",
                },
                "solver": {
                    "description": "The solver for weight optimization.",
                    "enum": ["lbfgs", "sgd", "adam"],
                    "default": "adam",
                },
                "alpha": {
                    "description": "L2 penalty (regularization term) parameter.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1,
                    "default": 0.0001,
                },
                "batch_size": {
                    "description": "Size of minibatches for stochastic optimizers.",
                    "anyOf": [
                        {
                            "description": "Size of minibatches",
                            "type": "integer",
                            "distribution": "uniform",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 128,
                        },
                        {
                            "description": "Automatic selection, batch_size=min(200, n_samples)",
                            "enum": ["auto"],
                        },
                    ],
                    "default": "auto",
                },
                "learning_rate": {
                    "description": "Learning rate schedule for weight updates.",
                    "enum": ["constant", "invscaling", "adaptive"],
                    "default": "constant",
                },
                "learning_rate_init": {
                    "description": "The initial learning rate used. It controls the "
                    "step-size in updating the weights.",
                    "type": "number",
                    "default": 0.001,
                },
                "power_t": {
                    "description": "The exponent for inverse scaling learning rate.",
                    "type": "number",
                    "default": 0.5,
                },
                "max_iter": {
                    "description": "Maximum number of iterations. The solver iterates until "
                    'convergence (determined by "tol") or this number of '
                    "iterations.",
                    "type": "integer",
                    "distribution": "uniform",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "default": 200,
                },
                "shuffle": {
                    "description": "Whether to shuffle samples in each iteration.",
                    "type": "boolean",
                    "default": True,
                },
                "random_state": {
                    "description": "Random generator selection",
                    "anyOf": [
                        {
                            "description": "seed used by the random number generators",
                            "type": "integer",
                        },
                        {
                            "description": "Random number generator",
                            "laleType": "numpy.random.RandomState",
                        },
                        {
                            "description": "RandomState instance used by np.random",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
                "tol": {
                    "description": "Tolerance for the optimization. When the loss or score "
                    "is not improving by at least tol for n_iter_no_change "
                    "consecutive iterations, unless learning_rate is set to "
                    '"adaptive", convergence is considered to be reached and '
                    "training stops.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "default": 0.0001,
                },
                "verbose": {
                    "description": "Whether to print progress messages to stdout.",
                    "type": "boolean",
                    "default": False,
                },
                "warm_start": {
                    "description": "When set to True, reuse the solution of the previous "
                    "call to fit as initialization, otherwise, just erase "
                    "the previous solution.",
                    "type": "boolean",
                    "default": False,
                },
                "momentum": {
                    "description": "Momentum for gradient descent update.",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.9,
                },
                "nesterovs_momentum": {
                    "description": "Whether to use Nesterov's momentum.",
                    "type": "boolean",
                    "default": True,
                },
                "early_stopping": {
                    "description": "Whether to use early stopping to terminate training when "
                    "validation score is not improving. If set to true, it "
                    "will automatically set aside 10% of training data as "
                    "validation and terminate training when validation score "
                    "is not improving by at least tol for n_iter_no_change "
                    "consecutive epochs.",
                    "type": "boolean",
                    "default": False,
                },
                "validation_fraction": {
                    "description": "The proportion of training data to set aside as "
                    "validation set for early stopping.",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                },
                "beta_1": {
                    "description": "Exponential decay rate for estimates of first moment "
                    "vector in adam.",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "exclusiveMaximum": True,
                    "default": 0.9,
                },
                "beta_2": {
                    "description": "Exponential decay rate for estimates of second moment "
                    "vector in adam.",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "exclusiveMaximum": True,
                    "default": 0.999,
                },
                "epsilon": {
                    "description": "Value for numerical stability in adam.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "default": 1e-08,
                },
                "n_iter_no_change": {
                    "description": "Maximum number of epochs to not meet tol improvement.",
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit the model to data matrix X and target(s) y.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
    },
}

_input_predict_schema = {
    "description": "Predict using the multi-layer perceptron classifier",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_predict_schema = {
    "description": "Predict using the multi-layer perceptron classifier",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_predict_proba_schema = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Multi-layer perceptron`_ dense deep neural network from scikit-learn for classification.

.. _`Multi-layer perceptron`: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.mlp_classifier.html",
    "import_from": "sklearn.neural_network",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["estimator", "classifier", "~interpretable"],
        "post": ["probabilities"],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}

MLPClassifier: lale.operators.IndividualOp
MLPClassifier = lale.operators.make_operator(MLPClassifierImpl, _combined_schemas)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.neural_network.MLPClassifier.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.neural_network.MLPClassifier.html
    from lale.schemas import Int

    MLPClassifier = MLPClassifier.customize_schema(
        max_fun=Int(
            desc="Maximum number of loss function calls.",
            default=15000,
            forOptimizer=False,
            min=0,
        )
    )

lale.docstrings.set_docstrings(MLPClassifierImpl, MLPClassifier._schemas)
