from numpy import inf, nan
from sklearn.linear_model import LogisticRegressionCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _LogisticRegressionCVImpl:
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
    "description": "inherited docstring for LogisticRegressionCV    Logistic Regression CV (aka logit, MaxEnt) classifier.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "Cs",
                "fit_intercept",
                "cv",
                "dual",
                "penalty",
                "scoring",
                "solver",
                "tol",
                "max_iter",
                "class_weight",
                "n_jobs",
                "verbose",
                "refit",
                "intercept_scaling",
                "multi_class",
                "random_state",
            ],
            "relevantToOptimizer": [
                "Cs",
                "fit_intercept",
                "cv",
                "dual",
                "penalty",
                "scoring",
                "solver",
                "tol",
                "max_iter",
                "multi_class",
            ],
            "additionalProperties": False,
            "properties": {
                "Cs": {
                    "XXX TODO XXX": "list of floats | int",
                    "description": "Each of the values in Cs describes the inverse of regularization strength",
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 11,
                    "distribution": "uniform",
                    "default": 10,
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Specifies if a constant (a.k.a",
                },
                "cv": {
                    "XXX TODO XXX": "integer or cross-validation generator, default: None",
                    "description": "The default cross-validation generator used is Stratified K-Folds",
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 4,
                    "distribution": "uniform",
                    "default": 3,
                },
                "dual": {
                    "type": "boolean",
                    "default": False,
                    "description": "Dual or primal formulation",
                },
                "penalty": {
                    "XXX TODO XXX": "str, 'l1' or 'l2'",
                    "description": "Used to specify the norm used in the penalization",
                    "enum": ["l1", "l2"],
                    "default": "l2",
                },
                "scoring": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": ["accuracy", None]},
                    ],
                    "default": None,
                    "description": "A string (see model evaluation documentation) or a scorer callable object / function with signature ``scorer(estimator, X, y)``",
                },
                "solver": {
                    "enum": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    "default": "lbfgs",
                    "description": "Algorithm to use in the optimization problem",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Tolerance for stopping criteria.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Maximum number of iterations of the optimization algorithm.",
                },
                "class_weight": {
                    "XXX TODO XXX": "dict or 'balanced', optional",
                    "description": "Weights associated with classes in the form ``{class_label: weight}``",
                    "enum": ["balanced"],
                    "default": "balanced",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "Number of CPU cores used during the cross-validation loop",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any positive number for verbosity.",
                },
                "refit": {
                    "type": "boolean",
                    "default": True,
                    "description": "If set to True, the scores are averaged across all folds, and the coefs and the C that corresponds to the best score is taken, and a final refit is done using these parameters",
                },
                "intercept_scaling": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True",
                },
                "multi_class": {
                    "enum": ["ovr", "multinomial", "auto"],
                    "default": "ovr",
                    "description": "If the option chosen is 'ovr', then a binary problem is fit for each label",
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
            },
        },
        {
            "XXX TODO XXX": "Parameter: dual > only implemented for l2 penalty with liblinear solver"
        },
        {"XXX TODO XXX": "Parameter: penalty > only l2 penalties"},
        {
            "XXX TODO XXX": "Parameter: solver > only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss; 'liblinear' is limited to one-versus-rest schemes"
        },
        {
            "XXX TODO XXX": "Parameter: intercept_scaling > only when the solver 'liblinear' is used and self"
        },
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
            "description": "Training vector, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target vector relative to X.",
        },
        "sample_weight": {
            "laleType": "Any",
            "XXX TODO XXX": "array-like, shape (n_samples,) optional",
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
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Probability estimates.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns the probability of the sample for each class in the model, where classes are ordered as they are in ``self.classes_``.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
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
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.LogisticRegressionCV#sklearn-linear_model-logisticregressioncv",
    "import_from": "sklearn.linear_model",
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
LogisticRegressionCV = make_operator(_LogisticRegressionCVImpl, _combined_schemas)

set_docstrings(LogisticRegressionCV)
