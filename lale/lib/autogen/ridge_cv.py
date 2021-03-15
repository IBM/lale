from numpy import inf, nan
from sklearn.linear_model import RidgeCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _RidgeCVImpl:
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
    "description": "inherited docstring for RidgeCV    Ridge regression with built-in cross-validation.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "alphas",
                "fit_intercept",
                "normalize",
                "scoring",
                "cv",
                "gcv_mode",
                "store_cv_values",
            ],
            "relevantToOptimizer": [
                "fit_intercept",
                "normalize",
                "scoring",
                "cv",
                "gcv_mode",
                "store_cv_values",
            ],
            "additionalProperties": False,
            "properties": {
                "alphas": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [0.1, 1.0, 10.0],
                    "description": "Array of alpha values to try",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model",
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False",
                },
                "scoring": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": ["accuracy", None]},
                    ],
                    "default": None,
                    "description": "A string (see model evaluation documentation) or a scorer callable object / function with signature ``scorer(estimator, X, y)``.",
                },
                "cv": {
                    "XXX TODO XXX": "int, cross-validation generator or an iterable, optional",
                    "description": "Determines the cross-validation splitting strategy",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 4,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "gcv_mode": {
                    "enum": [None, "auto", "svd", "eigen"],
                    "default": None,
                    "description": "Flag indicating which strategy to use when performing Generalized Cross-Validation",
                },
                "store_cv_values": {
                    "type": "boolean",
                    "default": False,
                    "description": "Flag indicating if the cross-validation values corresponding to each alpha should be stored in the ``cv_values_`` attribute (see below)",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: store_cv_values > only compatible with cv=none (i"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit Ridge regression model",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Target values",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "number"},
                {"type": "array", "items": {"type": "number"}},
            ],
            "description": "Sample weight",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array_like or sparse matrix, shape (n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Samples.",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns predicted values.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.RidgeCV#sklearn-linear_model-ridgecv",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
RidgeCV = make_operator(_RidgeCVImpl, _combined_schemas)

set_docstrings(RidgeCV)
