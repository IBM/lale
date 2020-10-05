from numpy import inf, nan
from sklearn.decomposition import FactorAnalysis as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class FactorAnalysisImpl:
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
    "description": "inherited docstring for FactorAnalysis    Factor Analysis (FA)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "tol",
                "copy",
                "max_iter",
                "noise_variance_init",
                "svd_method",
                "iterated_power",
                "random_state",
            ],
            "relevantToOptimizer": [
                "n_components",
                "tol",
                "copy",
                "max_iter",
                "svd_method",
                "iterated_power",
            ],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "enum": ["int", None],
                    "default": None,
                    "description": "Dimensionality of latent space, the number of components of ``X`` that are obtained after ``transform``",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.01,
                    "description": "Stopping tolerance for EM algorithm.",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to make a copy of X",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "Maximum number of iterations.",
                },
                "noise_variance_init": {
                    "XXX TODO XXX": "None | array, shape=(n_features,)",
                    "description": "The initial guess of the noise variance for each feature",
                    "enum": [None],
                    "default": None,
                },
                "svd_method": {
                    "enum": ["lapack", "randomized"],
                    "default": "randomized",
                    "description": "Which SVD method to use",
                },
                "iterated_power": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 4,
                    "distribution": "uniform",
                    "default": 3,
                    "description": "Number of iterations for the power method",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": 0,
                    "description": "If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: iterated_power > only used if svd_method equals 'randomized'"
        },
        {
            "XXX TODO XXX": "Parameter: random_state > only used when svd_method equals 'randomized'"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the FactorAnalysis model to X using EM",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply dimensionality reduction to X using the model.",
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
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The latent variables of X.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.FactorAnalysis#sklearn-decomposition-factoranalysis",
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
set_docstrings(FactorAnalysisImpl, _combined_schemas)
FactorAnalysis = make_operator(FactorAnalysisImpl, _combined_schemas)
