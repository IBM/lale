from numpy import inf, nan
from sklearn.linear_model import RidgeClassifierCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _RidgeClassifierCVImpl:
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

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for RidgeClassifierCV    Ridge classifier with built-in cross-validation.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "alphas",
                "fit_intercept",
                "normalize",
                "scoring",
                "cv",
                "class_weight",
                "store_cv_values",
            ],
            "relevantToOptimizer": ["fit_intercept", "normalize", "scoring", "cv"],
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
                "class_weight": {
                    "XXX TODO XXX": "dict or 'balanced', optional",
                    "description": "Weights associated with classes in the form ``{class_label: weight}``",
                    "enum": ["balanced"],
                    "default": "balanced",
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
    "description": "Fit the ridge classifier.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values",
        },
        "sample_weight": {
            "anyOf": [
                {"type": "number"},
                {"type": "array", "items": {"type": "number"}},
            ],
            "description": "Sample weight.",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict class labels for samples in X.",
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
    "description": "Predicted class label per sample.",
    "type": "array",
    "items": {"type": "number"},
}
_input_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict confidence scores for samples.",
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
_output_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Confidence scores per (sample, class) combination",
    "laleType": "Any",
    "XXX TODO XXX": "array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.RidgeClassifierCV#sklearn-linear_model-ridgeclassifiercv",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}
RidgeClassifierCV = make_operator(_RidgeClassifierCVImpl, _combined_schemas)

set_docstrings(RidgeClassifierCV)
