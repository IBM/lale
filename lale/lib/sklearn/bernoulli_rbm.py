from sklearn.neural_network import BernoulliRBM as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "BernoulliRBM: Bernoulli Restricted Boltzmann Machine (RBM).",
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
                    "minimumForOptimizer": 1e-3,
                    "maximumForOptimizer": 1.0,
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
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {
                            "description": "Use the provided random state, only affecting other users of that same random state instance.",
                            "laleType": "numpy.random.RandomState",
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                    "description": """Determines random number generation for:
Gibbs sampling from visible and hidden layers.
Initializing components, sampling from layers during fit.
Corrupting the data when scoring samples.
Pass an int for reproducible results across multiple function calls.""",
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
    "description": """`BernoulliRBM`_ : Bernoulli Restricted Boltzmann Machine (RBM).

.. _`BernoulliRBM: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.bernoulli_rbm.html",
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
BernoulliRBM = make_operator(Op, _combined_schemas)

set_docstrings(BernoulliRBM)
