from numpy import inf, nan
from sklearn.linear_model import RANSACRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class RANSACRegressorImpl:
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
    "description": "inherited docstring for RANSACRegressor    RANSAC (RANdom SAmple Consensus) algorithm.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
                "min_samples",
                "residual_threshold",
                "is_data_valid",
                "is_model_valid",
                "max_trials",
                "max_skips",
                "stop_n_inliers",
                "stop_score",
                "stop_probability",
                "loss",
                "random_state",
            ],
            "relevantToOptimizer": [
                "min_samples",
                "max_trials",
                "max_skips",
                "stop_n_inliers",
                "loss",
            ],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Base estimator object which implements the following methods:   * `fit(X, y)`: Fit model to given training data and target values",
                },
                "min_samples": {
                    "XXX TODO XXX": "int (>= 1) or float ([0, 1]), optional",
                    "description": "Minimum number of samples chosen randomly from original data",
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.0,
                            "maximumForOptimizer": 1.0,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "residual_threshold": {
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                    "description": "Maximum residual for a data sample to be classified as an inlier",
                },
                "is_data_valid": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "This function is called with the randomly selected data before the model is fitted to it: `is_data_valid(X, y)`",
                },
                "is_model_valid": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "This function is called with the estimated model and the randomly selected data: `is_model_valid(model, X, y)`",
                },
                "max_trials": {
                    "type": "integer",
                    "minimumForOptimizer": 100,
                    "maximumForOptimizer": 101,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Maximum number of iterations for random sample selection.",
                },
                "max_skips": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.0,
                            "maximumForOptimizer": 1.0,
                            "distribution": "uniform",
                        },
                    ],
                    "default": inf,
                    "description": "Maximum number of iterations that can be skipped due to finding zero inliers or invalid data defined by ``is_data_valid`` or invalid models defined by ``is_model_valid``",
                },
                "stop_n_inliers": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.0,
                            "maximumForOptimizer": 1.0,
                            "distribution": "uniform",
                        },
                    ],
                    "default": inf,
                    "description": "Stop iteration if at least this number of inliers are found.",
                },
                "stop_score": {
                    "type": "number",
                    "default": inf,
                    "description": "Stop iteration if score is greater equal than this threshold.",
                },
                "stop_probability": {
                    "XXX TODO XXX": "float in range [0, 1], optional",
                    "description": "RANSAC iteration stops if at least one outlier-free set of the training data is sampled in RANSAC",
                    "type": "number",
                    "default": 0.99,
                },
                "loss": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {
                            "enum": [
                                "X[i]",
                                "absolute_loss",
                                "deviance",
                                "epsilon_insensitive",
                                "exponential",
                                "hinge",
                                "huber",
                                "lad",
                                "linear",
                                "log",
                                "ls",
                                "modified_huber",
                                "perceptron",
                                "quantile",
                                "residual_threshold",
                                "square",
                                "squared_epsilon_insensitive",
                                "squared_hinge",
                                "squared_loss",
                            ]
                        },
                    ],
                    "default": "absolute_loss",
                    "description": 'String inputs, "absolute_loss" and "squared_loss" are supported which find the absolute loss and squared loss per sample respectively',
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The generator used to initialize the centers",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: base_estimator > only supports regression estimators"
        },
        {
            "XXX TODO XXX": "Parameter: is_model_valid > only be used if the estimated model is needed for making the rejection decision"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit estimator using RANSAC algorithm.",
    "type": "object",
    "required": ["X", "y"],
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
            "description": "Training data.",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Target values.",
        },
        "sample_weight": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Individual weights for each sample raises error if sample_weight is passed and base_estimator fit method does not support it.",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the estimated model.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.RANSACRegressor#sklearn-linear_model-ransacregressor",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
set_docstrings(RANSACRegressorImpl, _combined_schemas)
RANSACRegressor = make_operator(RANSACRegressorImpl, _combined_schemas)
