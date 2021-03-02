from numpy import inf, nan
from sklearn.ensemble import RandomTreesEmbedding as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _RandomTreesEmbeddingImpl:
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
    "description": "inherited docstring for RandomTreesEmbedding    An ensemble of totally random trees.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_impurity_split",
                "sparse_output",
                "n_jobs",
                "random_state",
                "verbose",
                "warm_start",
            ],
            "relevantToOptimizer": [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "sparse_output",
            ],
            "additionalProperties": False,
            "properties": {
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 10,
                    "description": "Number of trees in the forest",
                },
                "max_depth": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 5,
                    "distribution": "uniform",
                    "default": 5,
                    "description": "The maximum depth of each tree",
                },
                "min_samples_split": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.01,
                            "maximumForOptimizer": 0.5,
                            "distribution": "uniform",
                        },
                    ],
                    "default": 2,
                    "description": "The minimum number of samples required to split an internal node:  - If int, then consider `min_samples_split` as the minimum number",
                },
                "min_samples_leaf": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.01,
                            "maximumForOptimizer": 0.5,
                            "distribution": "uniform",
                        },
                    ],
                    "default": 1,
                    "description": "The minimum number of samples required to be at a leaf node",
                },
                "min_weight_fraction_leaf": {
                    "type": "number",
                    "default": 0.0,
                    "description": "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node",
                },
                "max_leaf_nodes": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Grow trees with ``max_leaf_nodes`` in best-first fashion",
                },
                "min_impurity_decrease": {
                    "type": "number",
                    "default": 0.0,
                    "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value",
                },
                "min_impurity_split": {
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                    "description": "Threshold for early stopping in tree growth",
                },
                "sparse_output": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not to return a sparse CSR matrix, as default behavior, or to return a dense array compatible with dense pipeline operators.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "The number of jobs to run in parallel for both `fit` and `predict`",
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
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Controls the verbosity when fitting and predicting.",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: min_samples_leaf > only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit estimator.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix, shape=(n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The input samples",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "description": "Sample weights",
        },
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform dataset.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix, shape=(n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Input data to be transformed",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed dataset.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.RandomTreesEmbedding#sklearn-ensemble-randomtreesembedding",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
RandomTreesEmbedding = make_operator(_RandomTreesEmbeddingImpl, _combined_schemas)

set_docstrings(RandomTreesEmbedding)
