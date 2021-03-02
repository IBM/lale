from numpy import inf, nan
from sklearn.svm import SVC as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _SVCImpl:
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

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for SVC    C-Support Vector Classification.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "C",
                "kernel",
                "degree",
                "gamma",
                "coef0",
                "shrinking",
                "probability",
                "tol",
                "cache_size",
                "class_weight",
                "verbose",
                "max_iter",
                "decision_function_shape",
                "random_state",
            ],
            "relevantToOptimizer": [
                "kernel",
                "degree",
                "gamma",
                "shrinking",
                "probability",
                "tol",
                "cache_size",
                "max_iter",
                "decision_function_shape",
            ],
            "additionalProperties": False,
            "properties": {
                "C": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Penalty parameter C of the error term.",
                },
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
                "shrinking": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use the shrinking heuristic.",
                },
                "probability": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to enable probability estimates",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.001,
                    "description": "Tolerance for stopping criterion.",
                },
                "cache_size": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 200,
                    "description": "Specify the size of the kernel cache (in MB).",
                },
                "class_weight": {
                    "enum": ["dict", "balanced"],
                    "default": "balanced",
                    "description": "Set the parameter C of class i to class_weight[i]*C for SVC",
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
                "decision_function_shape": {
                    "XXX TODO XXX": "'ovo', 'ovr', default='ovr'",
                    "description": "Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2)",
                    "enum": ["ovr"],
                    "default": "ovr",
                },
                "break_ties": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator used when shuffling the data for probability estimates",
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
    "description": "Perform classification on samples in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": 'For kernel="precomputed", the expected shape of X is [n_samples_test, n_samples_train]',
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Class labels for samples in X.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute probabilities of possible outcomes for samples in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": 'For kernel="precomputed", the expected shape of X is [n_samples_test, n_samples_train]',
        }
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns the probability of the sample for each class in the model",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_input_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Evaluates the decision function for the samples in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns the decision function of the sample for each class in the model",
    "laleType": "Any",
    "XXX TODO XXX": "array-like, shape (n_samples, n_classes * (n_classes-1) / 2)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.svm.SVC#sklearn-svm-svc",
    "import_from": "sklearn.svm",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}
SVC = make_operator(_SVCImpl, _combined_schemas)

set_docstrings(SVC)
