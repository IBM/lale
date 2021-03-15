from numpy import inf, nan
from sklearn.decomposition import PCA as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _PCAImpl:
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
    "description": "inherited docstring for PCA    Principal component analysis (PCA)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "copy",
                "whiten",
                "svd_solver",
                "tol",
                "iterated_power",
                "random_state",
            ],
            "relevantToOptimizer": [
                "n_components",
                "copy",
                "whiten",
                "svd_solver",
                "tol",
                "iterated_power",
            ],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.0,
                            "maximumForOptimizer": 1.0,
                            "distribution": "uniform",
                        },
                        {"type": "string", "forOptimizer": False},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Number of components to keep",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.",
                },
                "whiten": {
                    "type": "boolean",
                    "default": False,
                    "description": "When True (False by default) the `components_` vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances",
                },
                "svd_solver": {
                    "enum": ["auto", "full", "arpack", "randomized"],
                    "default": "auto",
                    "description": "auto :     the solver is selected by a default policy based on `X.shape` and     `n_components`: if the input data is larger than 500x500 and the     number of components to extract is lower than 80% of the smallest     dimension of the data, then the more efficient 'randomized'     method is enabled",
                },
                "tol": {
                    "XXX TODO XXX": "float >= 0, optional (default .0)",
                    "description": "Tolerance for singular values computed by svd_solver == 'arpack'",
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0,
                },
                "iterated_power": {
                    "XXX TODO XXX": "int >= 0, or 'auto', (default 'auto')",
                    "description": "Number of iterations for the power method computed by svd_solver == 'randomized'",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 4,
                            "distribution": "uniform",
                        },
                        {"enum": ["auto"]},
                    ],
                    "default": "auto",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model with X.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply dimensionality reduction to X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "New data, where n_samples is the number of samples and n_features is the number of features.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply dimensionality reduction to X.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.PCA#sklearn-decomposition-pca",
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
PCA = make_operator(_PCAImpl, _combined_schemas)

set_docstrings(PCA)
