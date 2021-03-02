from numpy import inf, nan
from sklearn.preprocessing import QuantileTransformer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _QuantileTransformerImpl:
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
    "description": "inherited docstring for QuantileTransformer    Transform features using quantiles information.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_quantiles",
                "output_distribution",
                "ignore_implicit_zeros",
                "subsample",
                "random_state",
                "copy",
            ],
            "relevantToOptimizer": [
                "n_quantiles",
                "output_distribution",
                "subsample",
                "copy",
            ],
            "additionalProperties": False,
            "properties": {
                "n_quantiles": {
                    "type": "integer",
                    "minimumForOptimizer": 1000,
                    "maximumForOptimizer": 1001,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "Number of quantiles to be computed",
                },
                "output_distribution": {
                    "enum": ["normal", "uniform"],
                    "default": "uniform",
                    "description": "Marginal distribution for the transformed data",
                },
                "ignore_implicit_zeros": {
                    "type": "boolean",
                    "default": False,
                    "description": "Only applies to sparse matrices",
                },
                "subsample": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 100000,
                    "distribution": "uniform",
                    "default": 100000,
                    "description": "Maximum number of samples used to estimate the quantiles for computational efficiency",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Set to False to perform inplace transformation and avoid a copy (if the input is already a numpy array).",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: ignore_implicit_zeros > only applies to sparse matrices"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the quantiles used for transforming.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "ndarray or sparse matrix, shape (n_samples, n_features)",
            "description": "The data used to scale along the features axis",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Feature-wise transformation of the data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "ndarray or sparse matrix, shape (n_samples, n_features)",
            "description": "The data used to scale along the features axis",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The projected data.",
    "laleType": "Any",
    "XXX TODO XXX": "ndarray or sparse matrix, shape (n_samples, n_features)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.QuantileTransformer#sklearn-preprocessing-quantiletransformer",
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
QuantileTransformer = make_operator(_QuantileTransformerImpl, _combined_schemas)

set_docstrings(QuantileTransformer)
