from numpy import inf, nan
from sklearn.kernel_approximation import Nystroem as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _NystroemImpl:
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
    "description": "inherited docstring for Nystroem    Approximate a kernel map using a subset of the training data.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "kernel",
                "gamma",
                "coef0",
                "degree",
                "kernel_params",
                "n_components",
                "random_state",
            ],
            "relevantToOptimizer": ["kernel", "n_components"],
            "additionalProperties": False,
            "properties": {
                "kernel": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": ["linear", "poly", "rbf", "sigmoid"]},
                    ],
                    "default": "rbf",
                    "description": "Kernel map to be approximated",
                },
                "gamma": {
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                    "description": "Gamma parameter for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels",
                },
                "coef0": {
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                    "description": "Zero coefficient for polynomial and sigmoid kernels",
                },
                "degree": {
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                    "description": "Degree of the polynomial kernel",
                },
                "kernel_params": {
                    "XXX TODO XXX": "mapping of string to any, optional",
                    "description": "Additional parameters (keyword arguments) for kernel function passed as callable object.",
                    "enum": [None],
                    "default": None,
                },
                "n_components": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 256,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Number of features to construct",
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
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit estimator to data.",
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
    "description": "Apply feature map to X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Data to transform.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.kernel_approximation.Nystroem#sklearn-kernel_approximation-nystroem",
    "import_from": "sklearn.kernel_approximation",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
Nystroem = make_operator(_NystroemImpl, _combined_schemas)

set_docstrings(Nystroem)
