from numpy import inf, nan
from sklearn.decomposition import KernelPCA as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _KernelPCAImpl:
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
    "description": "inherited docstring for KernelPCA    Kernel Principal component analysis (KPCA)",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "kernel",
                "gamma",
                "degree",
                "coef0",
                "kernel_params",
                "alpha",
                "fit_inverse_transform",
                "eigen_solver",
                "tol",
                "max_iter",
                "remove_zero_eig",
                "random_state",
                "copy_X",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "n_components",
                "kernel",
                "degree",
                "coef0",
                "alpha",
                "eigen_solver",
                "tol",
                "max_iter",
                "remove_zero_eig",
                "copy_X",
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
                    "description": "Number of components",
                },
                "kernel": {
                    "enum": [
                        "linear",
                        "poly",
                        "rbf",
                        "sigmoid",
                        "cosine",
                        "precomputed",
                    ],
                    "default": "linear",
                    "description": "Kernel",
                },
                "gamma": {
                    "XXX TODO XXX": "float, default=1/n_features",
                    "description": "Kernel coefficient for rbf, poly and sigmoid kernels",
                    "enum": [None],
                    "default": None,
                },
                "degree": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 3,
                    "distribution": "uniform",
                    "default": 3,
                    "description": "Degree for poly kernels",
                },
                "coef0": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 1,
                    "description": "Independent term in poly and sigmoid kernels",
                },
                "kernel_params": {
                    "XXX TODO XXX": "mapping of string to any, default=None",
                    "description": "Parameters (keyword arguments) and values for kernel passed as callable object",
                    "enum": [None],
                    "default": None,
                },
                "alpha": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 1e-10,
                            "maximumForOptimizer": 1.0,
                            "distribution": "loguniform",
                        },
                    ],
                    "default": 1.0,
                    "description": "Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).",
                },
                "fit_inverse_transform": {
                    "type": "boolean",
                    "default": False,
                    "description": "Learn the inverse transform for non-precomputed kernels",
                },
                "eigen_solver": {
                    "enum": ["auto", "dense", "arpack"],
                    "default": "auto",
                    "description": "Select eigensolver to use",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0,
                    "description": "Convergence tolerance for arpack",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 10,
                            "maximumForOptimizer": 1000,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Maximum number of iterations for arpack",
                },
                "remove_zero_eig": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, then all components with zero eigenvalues are removed, so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability)",
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
                "copy_X": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, input X is copied and stored by the model in the `X_fit_` attribute",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "The number of parallel jobs to run",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model from data in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vector, where n_samples in the number of samples and n_features is the number of features.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.KernelPCA#sklearn-decomposition-kernelpca",
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
KernelPCA = make_operator(_KernelPCAImpl, _combined_schemas)

set_docstrings(KernelPCA)
