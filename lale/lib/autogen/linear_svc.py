from numpy import inf, nan
from sklearn.svm import LinearSVC as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _LinearSVCImpl:
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
    "description": "inherited docstring for LinearSVC    Linear Support Vector Classification.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "penalty",
                "loss",
                "dual",
                "tol",
                "C",
                "multi_class",
                "fit_intercept",
                "intercept_scaling",
                "class_weight",
                "verbose",
                "random_state",
                "max_iter",
            ],
            "relevantToOptimizer": [
                "penalty",
                "loss",
                "dual",
                "tol",
                "multi_class",
                "fit_intercept",
                "intercept_scaling",
                "max_iter",
            ],
            "additionalProperties": False,
            "properties": {
                "penalty": {
                    "XXX TODO XXX": "string, 'l1' or 'l2' (default='l2')",
                    "description": "Specifies the norm used in the penalization",
                    "enum": ["l1", "l2"],
                    "default": "l2",
                },
                "loss": {
                    "XXX TODO XXX": "string, 'hinge' or 'squared_hinge' (default='squared_hinge')",
                    "description": "Specifies the loss function",
                    "enum": [
                        "epsilon_insensitive",
                        "hinge",
                        "squared_epsilon_insensitive",
                        "squared_hinge",
                    ],
                    "default": "squared_hinge",
                },
                "dual": {
                    "type": "boolean",
                    "default": True,
                    "description": "Select the algorithm to either solve the dual or primal optimization problem",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance for stopping criteria.",
                },
                "C": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Penalty parameter C of the error term.",
                },
                "multi_class": {
                    "XXX TODO XXX": "string, 'ovr' or 'crammer_singer' (default='ovr')",
                    "description": "Determines the multi-class strategy if `y` contains more than two classes",
                    "enum": ["crammer_singer", "ovr"],
                    "default": "ovr",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model",
                },
                "intercept_scaling": {
                    "type": "number",
                    "minimumForOptimizer": 0.0,
                    "maximumForOptimizer": 1.0,
                    "distribution": "uniform",
                    "default": 1,
                    "description": "When self.fit_intercept is True, instance vector x becomes ``[x, self.intercept_scaling]``, i.e",
                },
                "class_weight": {
                    "enum": ["dict", "balanced"],
                    "default": "balanced",
                    "description": "Set the parameter C of class i to ``class_weight[i]*C`` for SVC",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Enable verbose output",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator to use when shuffling the data for the dual coordinate descent (if ``dual=True``)",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "The maximum number of iterations to be run.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model according to the given training data.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vector, where n_samples in the number of samples and n_features is the number of features.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target vector relative to X",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Array of weights that are assigned to individual samples",
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.svm.LinearSVC#sklearn-svm-linearsvc",
    "import_from": "sklearn.svm",
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
LinearSVC = make_operator(_LinearSVCImpl, _combined_schemas)

set_docstrings(LinearSVC)
