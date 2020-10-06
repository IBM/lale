from numpy import inf, nan
from sklearn.ensemble import RandomForestRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class RandomForestRegressorImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for RandomForestRegressor    A random forest regressor.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_estimators",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_impurity_split",
                "bootstrap",
                "oob_score",
                "n_jobs",
                "random_state",
                "verbose",
                "warm_start",
            ],
            "relevantToOptimizer": [
                "n_estimators",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "bootstrap",
            ],
            "additionalProperties": False,
            "properties": {
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 10,
                    "description": "The number of trees in the forest",
                },
                "criterion": {
                    "enum": ["friedman_mse", "mse"],
                    "default": "mse",
                    "description": "The function to measure the quality of a split",
                },
                "max_depth": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 5,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The maximum depth of the tree",
                },
                "min_samples_split": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 5,
                            "distribution": "uniform",
                        },
                        {
                            "type": "number",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 5,
                            "distribution": "uniform",
                        },
                    ],
                    "default": 2,
                    "description": "The minimum number of samples required to split an internal node:  - If int, then consider `min_samples_split` as the minimum number",
                },
                "min_samples_leaf": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 1,
                            "maximumForOptimizer": 5,
                            "distribution": "uniform",
                        },
                        {
                            "type": "number",
                            "minimumForOptimizer": 1,
                            "maximumForOptimizer": 5,
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
                "max_features": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.01,
                            "maximumForOptimizer": 1.0,
                            "distribution": "uniform",
                        },
                        {"type": "string", "forOptimizer": False},
                        {"enum": [None]},
                    ],
                    "default": "auto",
                    "description": "The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split",
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
                "bootstrap": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether bootstrap samples are used when building trees",
                },
                "oob_score": {
                    "type": "boolean",
                    "default": False,
                    "description": "whether to use out-of-bag samples to estimate the R^2 on unseen data.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 4,
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
    "description": "Build a forest of trees from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix of shape = [n_samples, n_features]",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The training input samples",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The target values (class labels in classification, real numbers in regression).",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "description": "Sample weights",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict regression target for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix of shape = [n_samples, n_features]",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The input samples",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.RandomForestRegressor#sklearn-ensemble-randomforestregressor",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
set_docstrings(RandomForestRegressorImpl, _combined_schemas)
RandomForestRegressor = make_operator(RandomForestRegressorImpl, _combined_schemas)
