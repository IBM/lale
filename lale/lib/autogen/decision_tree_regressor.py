import sklearn
from numpy import inf, nan
from sklearn.tree import DecisionTreeRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _DecisionTreeRegressorImpl:
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
    "description": "inherited docstring for DecisionTreeRegressor    A decision tree regressor.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "random_state",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_impurity_split",
                "presort",
            ],
            "relevantToOptimizer": [
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
            ],
            "additionalProperties": False,
            "properties": {
                "criterion": {
                    "enum": ["friedman_mse", "mse"],
                    "default": "mse",
                    "description": "The function to measure the quality of a split",
                },
                "splitter": {
                    "enum": ["random", "best"],
                    "default": "best",
                    "description": "The strategy used to choose the split at each node",
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
                    "default": None,
                    "description": "The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split",
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
                "max_leaf_nodes": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Grow a tree with ``max_leaf_nodes`` in best-first fashion",
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
                "presort": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to presort the data to speed up the finding of best splits in fitting",
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
    "description": "Build a decision tree regressor from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix, shape = [n_samples, n_features]",
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
            "description": "The target values (real numbers)",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "description": "Sample weights",
        },
        "check_input": {
            "type": "boolean",
            "default": True,
            "description": "Allow to bypass several input checking",
        },
        "X_idx_sorted": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "The indexes of the sorted training input samples",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict class or regression value for X.",
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
        },
        "check_input": {
            "type": "boolean",
            "default": True,
            "description": "Allow to bypass several input checking",
        },
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted classes, or the predict values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.tree.DecisionTreeRegressor#sklearn-tree-decisiontreeregressor",
    "import_from": "sklearn.tree",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
DecisionTreeRegressor = make_operator(_DecisionTreeRegressorImpl, _combined_schemas)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # new: https://scikit-learn.org/0.22/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    from lale.schemas import AnyOf, Bool, Enum, Float

    DecisionTreeRegressor = DecisionTreeRegressor.customize_schema(
        presort=AnyOf(
            types=[Bool(), Enum(["deprecated"])],
            desc="This parameter is deprecated and will be removed in v0.24.",
            default="deprecated",
        ),
        ccp_alpha=Float(
            desc="Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.",
            default=0.0,
            forOptimizer=False,
            minimum=0.0,
            maximumForOptimizer=0.1,
        ),
        set_as_available=True,
    )

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.22/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    DecisionTreeRegressor = DecisionTreeRegressor.customize_schema(
        criterion={
            "description": "Function to measure the quality of a split.",
            "anyOf": [
                {"enum": ["mse", "friedman_mse", "poisson"]},
                {"enum": ["mae"], "forOptimizer": False},
            ],
            "default": "mse",
        },
        presort=None,
        set_as_available=True,
    )

if sklearn.__version__ >= "1.0":
    # old: https://scikit-learn.org/0.24/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # new: https://scikit-learn.org/1.0/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    DecisionTreeRegressor = DecisionTreeRegressor.customize_schema(
        criterion={
            "description": "Function to measure the quality of a split.",
            "anyOf": [
                {
                    "enum": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },
                {"enum": ["mae", "mse"], "forOptimizer": False},
            ],
            "default": "squared_error",
        },
        min_impurity_split=None,
        set_as_available=True,
    )

set_docstrings(DecisionTreeRegressor)
