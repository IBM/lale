from numpy import inf, nan
from sklearn.decomposition import SparsePCA as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class SparsePCAImpl:
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
    "description": "inherited docstring for SparsePCA    Sparse Principal Components Analysis (SparsePCA)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "alpha",
                "ridge_alpha",
                "max_iter",
                "tol",
                "method",
                "n_jobs",
                "U_init",
                "V_init",
                "verbose",
                "random_state",
            ],
            "relevantToOptimizer": [
                "n_components",
                "alpha",
                "max_iter",
                "tol",
                "method",
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
                    "description": "Number of sparse atoms to extract.",
                },
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1,
                    "description": "Sparsity controlling parameter",
                },
                "ridge_alpha": {
                    "type": "number",
                    "default": 0.01,
                    "description": "Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "Maximum number of iterations to perform.",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 1e-08,
                    "description": "Tolerance for the stopping condition.",
                },
                "method": {
                    "enum": ["lars", "cd"],
                    "default": "lars",
                    "description": "lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso)",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "Number of parallel jobs to run",
                },
                "U_init": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Initial values for the loadings for warm restart scenarios.",
                },
                "V_init": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Initial values for the components for warm restart scenarios.",
                },
                "verbose": {
                    "anyOf": [{"type": "integer"}, {"type": "boolean"}],
                    "default": False,
                    "description": "Controls the verbosity; the higher, the more messages",
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
    "description": "Fit the model from data in X.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vector, where n_samples in the number of samples and n_features is the number of features.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Least Squares projection of the data onto the sparse components.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Test data to be transformed, must have the same number of features as the data used to train the model.",
        },
        "ridge_alpha": {
            "type": "number",
            "default": 0.01,
            "description": "Amount of ridge shrinkage to apply in order to improve conditioning",
        },
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed data.",
    "laleType": "Any",
    "XXX TODO XXX": "X_new array, shape (n_samples, n_components)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.SparsePCA#sklearn-decomposition-sparsepca",
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
set_docstrings(SparsePCAImpl, _combined_schemas)
SparsePCA = make_operator(SparsePCAImpl, _combined_schemas)
