from numpy import inf, nan
from sklearn.svm import SVR as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _SVRImpl:
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
    "description": "inherited docstring for SVR    Epsilon-Support Vector Regression.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "kernel",
                "degree",
                "gamma",
                "coef0",
                "tol",
                "C",
                "epsilon",
                "shrinking",
                "cache_size",
                "verbose",
                "max_iter",
            ],
            "relevantToOptimizer": [
                "kernel",
                "degree",
                "gamma",
                "tol",
                "epsilon",
                "shrinking",
                "cache_size",
                "max_iter",
            ],
            "additionalProperties": False,
            "properties": {
                "kernel": {
                    "enum": ["linear", "poly", "precomputed", "sigmoid", "rbf"],
                    "default": "rbf",
                    "description": "Specifies the kernel type to be used in the algorithm",
                },
                "degree": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 3,
                    "distribution": "uniform",
                    "default": 3,
                    "description": "Degree of the polynomial kernel function ('poly')",
                },
                "gamma": {
                    "anyOf": [
                        {"type": "number", "forOptimizer": False},
                        {"enum": ["scale", "auto"]},
                    ],
                    "default": "scale",
                    "description": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'",
                },
                "coef0": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Independent term in kernel function",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.001,
                    "description": "Tolerance for stopping criterion.",
                },
                "C": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Penalty parameter C of the error term.",
                },
                "epsilon": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 1.35,
                    "distribution": "loguniform",
                    "default": 0.1,
                    "description": "Epsilon in the epsilon-SVR model",
                },
                "shrinking": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use the shrinking heuristic.",
                },
                "cache_size": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 200,
                    "description": "Specify the size of the kernel cache (in MB).",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable verbose output",
                },
                "max_iter": {
                    "XXX TODO XXX": "int, optional (default=-1)",
                    "description": "Hard limit on iterations within solver, or -1 for no limit.",
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": (-1),
                },
            },
        },
        {"XXX TODO XXX": "Parameter: coef0 > only significant in 'poly' and 'sigmoid'"},
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the SVM model according to the given training data.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values (class labels in classification, real numbers in regression)",
        },
        "sample_weight": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Per-sample weights",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Perform regression on samples in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": 'For kernel="precomputed", the expected shape of X is (n_samples_test, n_samples_train).',
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Perform regression on samples in X.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.svm.SVR#sklearn-svm-svr",
    "import_from": "sklearn.svm",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
SVR = make_operator(_SVRImpl, _combined_schemas)

set_docstrings(SVR)
