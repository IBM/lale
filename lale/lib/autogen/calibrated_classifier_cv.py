import sklearn
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
                    "enum": ["sigmoid", "isotonic"],
                    "default": "sigmoid",
                },
                "cv": {
                    "description": """Cross-validation as integer or as object that has a split function.
                        The fit method performs cross validation on the input dataset for per
                        trial, and uses the mean cross validation performance for optimization.
                        This behavior is also impacted by handle_cv_failure flag.
                        If integer: number of folds in sklearn.model_selection.StratifiedKFold.
                        If object with split function: generator yielding (train, test) splits
                        as arrays of indices. Can use any of the iterators from
                        https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators.""",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "default": 5,
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 4,
                            "distribution": "uniform",
                        },
                        {"laleType": "Any", "forOptimizer": False},
                        {"enum": [None]},
                        {"enum": ["prefit"]},
                    ],
                    "default": None,
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
CalibratedClassifierCV = make_operator(_CalibratedClassifierCVImpl, _combined_schemas)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.calibration.CalibratedClassifierCV#sklearn-calibration-calibratedclassifiercv
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.calibration.CalibratedClassifierCV#sklearn-calibration-calibratedclassifiercv
    CalibratedClassifierCV = CalibratedClassifierCV.customize_schema(
        n_jobs={
            "description": "Number of jobs to run in parallel.",
            "anyOf": [
                {
                    "description": "1 unless in joblib.parallel_backend context.",
                    "enum": [None],
                },
                {"description": "Use all processors.", "enum": [-1]},
                {
                    "description": "Number of jobs to run in parallel.",
                    "type": "integer",
                    "minimum": 1,
                },
            ],
            "default": None,
        },
        set_as_available=True,
    )
    CalibratedClassifierCV = CalibratedClassifierCV.customize_schema(
        ensemble={
            "type": "boolean",
            "default": True,
            "description": "Determines how the calibrator is fitted when cv is not 'prefit'. Ignored if cv='prefit",
        },
        set_as_available=True,
    )


set_docstrings(CalibratedClassifierCV)
