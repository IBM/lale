from numpy import inf, nan
from sklearn.ensemble import AdaBoostClassifier as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _AdaBoostClassifierImpl:
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
    "description": "inherited docstring for AdaBoostClassifier    An AdaBoost classifier.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
                "n_estimators",
                "learning_rate",
                "algorithm",
                "random_state",
            ],
            "relevantToOptimizer": ["n_estimators", "algorithm"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "The base estimator from which the boosted ensemble is built",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "distribution": "uniform",
                    "default": 50,
                    "description": "The maximum number of estimators at which boosting is terminated",
                },
                "learning_rate": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Learning rate shrinks the contribution of each classifier by ``learning_rate``",
                },
                "algorithm": {
                    "enum": ["SAMME", "SAMME.R"],
                    "default": "SAMME.R",
                    "description": "If 'SAMME.R' then use the SAMME.R real boosting algorithm",
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
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Build a boosted classifier from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The training input samples",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The target values (class labels).",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Sample weights",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict classes for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The training input samples",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted classes.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict class probabilities for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The training input samples",
        }
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The class probabilities of the input samples",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_input_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the decision function of ``X``.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The training input samples",
        }
    },
}
_output_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The decision function of the input samples",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.AdaBoostClassifier#sklearn-ensemble-adaboostclassifier",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
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
AdaBoostClassifier = make_operator(_AdaBoostClassifierImpl, _combined_schemas)

set_docstrings(AdaBoostClassifier)
