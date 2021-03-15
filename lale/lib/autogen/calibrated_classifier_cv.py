from numpy import inf, nan
from sklearn.calibration import CalibratedClassifierCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _CalibratedClassifierCVImpl:
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
    "description": "inherited docstring for CalibratedClassifierCV    Probability calibration with isotonic regression or sigmoid.",
    "allOf": [
        {
            "type": "object",
            "required": ["base_estimator", "method", "cv"],
            "relevantToOptimizer": ["method", "cv"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "XXX TODO XXX": "instance BaseEstimator",
                    "description": "The classifier whose output decision function needs to be calibrated to offer more accurate predict_proba outputs",
                    "enum": [None],
                    "default": None,
                },
                "method": {
                    "XXX TODO XXX": "'sigmoid' or 'isotonic'",
                    "description": "The method to use for calibration",
                    "enum": ["sigmoid"],
                    "default": "sigmoid",
                },
                "cv": {
                    "XXX TODO XXX": 'integer, cross-validation generator, iterable or "prefit", optional',
                    "description": "Determines the cross-validation splitting strategy",
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 4,
                    "distribution": "uniform",
                    "default": 3,
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the calibrated model",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values.",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "description": "Sample weights",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict the target of new samples. Can be different from the",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The samples.",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted class.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Posterior probabilities of classification",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The samples.",
        }
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The predicted probas.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.calibration.CalibratedClassifierCV#sklearn-calibration-calibratedclassifiercv",
    "import_from": "sklearn.calibration",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}
CalibratedClassifierCV = make_operator(_CalibratedClassifierCVImpl, _combined_schemas)

set_docstrings(CalibratedClassifierCV)
