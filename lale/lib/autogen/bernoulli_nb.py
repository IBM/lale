from numpy import inf, nan
from sklearn.naive_bayes import BernoulliNB as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _BernoulliNBImpl:
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


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for BernoulliNB    Naive Bayes classifier for multivariate Bernoulli models.",
    "allOf": [
        {
            "type": "object",
            "required": ["alpha", "binarize", "fit_prior", "class_prior"],
            "relevantToOptimizer": ["alpha", "fit_prior", "binarize"],
            "additionalProperties": False,
            "properties": {
                "alpha": {
                    "type": "number",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1.0,
                    "description": "Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).",
                },
                "binarize": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": -1.0,
                            "maximumForOptimizer": 1.0,
                        },
                        {"enum": [None]},
                    ],
                    "default": 0.0,
                    "description": "Threshold for binarizing (mapping to booleans) of sample features",
                },
                "fit_prior": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to learn class prior probabilities or not",
                },
                "class_prior": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Prior probabilities of the classes",
                },
            },
        },
        {
            "description": "Cannot binarize a sparse matrix with threshold < 0",
            "anyOf": [
                {"type": "object", "properties": {"binarize": {"enum": [None]}}},
                {"type": "object", "laleNot": "X/isSparse"},
                {
                    "type": "object",
                    "properties": {"binarize": {"type": "number", "minimum": 0}},
                },
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit Naive Bayes classifier according to X, y",
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
            "description": "Target values.",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Weights applied to individual samples (1",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Perform classification on an array of test vectors X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predicted target values for X",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Return probability estimates for the test vector X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns the probability of the samples for each class in the model",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.naive_bayes.BernoulliNB#sklearn-naive_bayes-bernoullinb",
    "import_from": "sklearn.naive_bayes",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}
BernoulliNB = make_operator(_BernoulliNBImpl, _combined_schemas)

set_docstrings(BernoulliNB)
