from numpy import inf, nan
from sklearn.manifold import Isomap as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class IsomapImpl:
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
    "description": "inherited docstring for Isomap    Isomap Embedding",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_neighbors",
                "n_components",
                "eigen_solver",
                "tol",
                "max_iter",
                "path_method",
                "neighbors_algorithm",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "n_neighbors",
                "n_components",
                "eigen_solver",
                "tol",
                "path_method",
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
                "eigen_solver": {
                    "enum": ["auto", "arpack", "dense"],
                    "default": "auto",
                    "description": "'auto' : Attempt to choose the most efficient solver for the given problem",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                    "distribution": "uniform",
                    "default": 0,
                    "description": "Convergence tolerance passed to arpack or lobpcg",
                },
                "max_iter": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Maximum number of iterations for the arpack solver",
                },
                "path_method": {
                    "enum": ["auto", "FW", "D"],
                    "default": "auto",
                    "description": "Method to use in finding shortest path",
                },
                "neighbors_algorithm": {
                    "enum": ["auto", "brute", "kd_tree", "ball_tree"],
                    "default": "auto",
                    "description": "Algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance.",
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
    "description": "Compute the embedding vectors for data X",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
            "XXX TODO XXX": "{array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}",
            "description": "Sample data, shape = (n_samples, n_features), in the form of a numpy array, precomputed tree, or NearestNeighbors object.",
        },
        "y": {},
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.manifold.Isomap#sklearn-manifold-isomap",
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
set_docstrings(IsomapImpl, _combined_schemas)
Isomap = make_operator(IsomapImpl, _combined_schemas)
