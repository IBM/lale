from numpy import inf, nan
from sklearn.preprocessing import PolynomialFeatures as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _PolynomialFeaturesImpl:
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
    "description": "inherited docstring for PolynomialFeatures    Generate polynomial and interaction features.",
    "allOf": [
        {
            "type": "object",
            "required": ["degree", "interaction_only", "include_bias"],
            "relevantToOptimizer": ["degree", "interaction_only"],
            "additionalProperties": False,
            "properties": {
                "degree": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 3,
                    "distribution": "uniform",
                    "default": 2,
                    "description": "The degree of the polynomial features",
                },
                "interaction_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, only interaction features are produced: features that are products of at most ``degree`` *distinct* input features (so not ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).",
                },
                "include_bias": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True (default), then include a bias column, the feature in which all polynomial powers are zero (i.e",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: interaction_only > only interaction features are produced: features that are products of at most degree *distinct* input features (so not x[1] ** 2, x[0] * x[2] ** 3, etc"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute number of output features.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform data to polynomial features",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix, shape [n_samples, n_features]",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The data to transform, row by row",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The matrix of features, where NP is the number of polynomial features generated from the combination of inputs.",
    "laleType": "Any",
    "XXX TODO XXX": "np.ndarray or CSC sparse matrix, shape [n_samples, NP]",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.PolynomialFeatures#sklearn-preprocessing-polynomialfeatures",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
PolynomialFeatures = make_operator(_PolynomialFeaturesImpl, _combined_schemas)

set_docstrings(PolynomialFeatures)
