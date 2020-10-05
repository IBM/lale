from numpy import inf, nan
from sklearn.manifold import LocallyLinearEmbedding as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class LocallyLinearEmbeddingImpl:
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
    "description": "inherited docstring for LocallyLinearEmbedding    Locally Linear Embedding",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_neighbors",
                "n_components",
                "reg",
                "eigen_solver",
                "tol",
                "max_iter",
                "method",
                "hessian_tol",
                "modified_tol",
                "neighbors_algorithm",
                "random_state",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "n_neighbors",
                "n_components",
                "eigen_solver",
                "tol",
                "max_iter",
                "method",
                "neighbors_algorithm",
            ],
            "additionalProperties": False,
            "properties": {
                "n_neighbors": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 20,
                    "distribution": "uniform",
                    "default": 5,
                    "description": "number of neighbors to consider for each point.",
                },
                "n_components": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 256,
                    "distribution": "uniform",
                    "default": 2,
                    "description": "number of coordinates for the manifold",
                },
                "reg": {
                    "type": "number",
                    "default": 0.001,
                    "description": "regularization constant, multiplies the trace of the local covariance matrix of the distances.",
                },
                "eigen_solver": {
                    "enum": ["auto", "arpack", "dense"],
                    "default": "auto",
                    "description": "auto : algorithm will attempt to choose the best method for input data  arpack : use arnoldi iteration in shift-invert mode",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 1e-06,
                    "description": "Tolerance for 'arpack' method Not used if eigen_solver=='dense'.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "maximum number of iterations for the arpack solver",
                },
                "method": {
                    "XXX TODO XXX": "string ('standard', 'hessian', 'modified' or 'ltsa')",
                    "description": "standard : use the standard locally linear embedding algorithm",
                    "enum": ["ltsa", "modified", "standard"],
                    "default": "standard",
                },
                "hessian_tol": {
                    "type": "number",
                    "default": 0.0001,
                    "description": "Tolerance for Hessian eigenmapping method",
                },
                "modified_tol": {
                    "type": "number",
                    "default": 1e-12,
                    "description": "Tolerance for modified LLE method",
                },
                "neighbors_algorithm": {
                    "enum": ["auto", "brute", "kd_tree", "ball_tree"],
                    "default": "auto",
                    "description": "algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance",
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
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "The number of parallel jobs to run",
                },
            },
        },
        {
            "description": "hessian_tol, only used if method == 'hessian'",
            "anyOf": [
                {"type": "object", "properties": {"hessian_tol": {"enum": [0.0001]}}},
                {"type": "object", "properties": {"method": {"enum": ["hessian"]}}},
            ],
        },
        {
            "description": "modified_tol, only used if method == 'modified'",
            "anyOf": [
                {"type": "object", "properties": {"modified_tol": {"enum": [1e-12]}}},
                {"type": "object", "properties": {"method": {"enum": ["modified"]}}},
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the embedding vectors for data X",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "training set.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform new points into embedding space.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform new points into embedding space.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.manifold.LocallyLinearEmbedding#sklearn-manifold-locallylinearembedding",
    "import_from": "sklearn.manifold",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
set_docstrings(LocallyLinearEmbeddingImpl, _combined_schemas)
LocallyLinearEmbedding = make_operator(LocallyLinearEmbeddingImpl, _combined_schemas)
