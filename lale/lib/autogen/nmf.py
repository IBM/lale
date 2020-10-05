from numpy import inf, nan
from sklearn.decomposition import NMF as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class NMFImpl:
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
    "description": "inherited docstring for NMF    Non-Negative Matrix Factorization (NMF)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "init",
                "solver",
                "beta_loss",
                "tol",
                "max_iter",
                "random_state",
                "alpha",
                "l1_ratio",
                "verbose",
                "shuffle",
            ],
            "relevantToOptimizer": [
                "n_components",
                "tol",
                "max_iter",
                "alpha",
                "shuffle",
            ],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 256,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Number of components, if n_components is not set all features are kept.",
                },
                "init": {
                    "enum": ["custom", "nndsvd", "nndsvda", "nndsvdar", "random", None],
                    "default": None,
                    "description": "Method used to initialize the procedure",
                },
                "solver": {
                    "enum": ["cd", "mu"],
                    "default": "cd",
                    "description": "Numerical solver to use: 'cd' is a Coordinate Descent solver",
                },
                "beta_loss": {
                    "anyOf": [{"type": "number"}, {"type": "string"}],
                    "default": "frobenius",
                    "description": "String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance of the stopping condition.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 200,
                    "description": "Maximum number of iterations before timing out.",
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
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 0.0,
                    "description": "Constant that multiplies the regularization terms",
                },
                "l1_ratio": {
                    "type": "number",
                    "default": 0.0,
                    "description": "The regularization mixing parameter, with 0 <= l1_ratio <= 1",
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": 0,
                    "description": "Whether to be verbose.",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, randomize the order of coordinates in the CD solver",
                },
            },
        },
        {
            "description": "beta_loss, only in 'mu' solver",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"beta_loss": {"enum": ["frobenius"]}},
                },
                {"type": "object", "properties": {"solver": {"enum": ["mu"]}}},
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Learn a NMF model for the data X.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Data matrix to be decomposed",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform the data X according to the fitted NMF model",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Data matrix to be transformed by the model",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed data",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.NMF#sklearn-decomposition-nmf",
    "import_from": "sklearn.decomposition",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
set_docstrings(NMFImpl, _combined_schemas)
NMF = make_operator(NMFImpl, _combined_schemas)
