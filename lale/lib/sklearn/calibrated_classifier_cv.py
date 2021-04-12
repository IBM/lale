import typing

import sklearn
from sklearn.calibration import CalibratedClassifierCV as Op

import lale.operators
from lale.docstrings import set_docstrings
from lale.operators import make_operator
from lale.schemas import AnyOf, Bool, Int, Null

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Probability calibration with isotonic regression or sigmoid.",
    "allOf": [
        {
            "type": "object",
            "required": ["base_estimator", "method", "cv"],
            "relevantToOptimizer": ["method", "cv"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "description": "The classifier whose output decision function needs to be calibrated to offer more accurate predict_proba outputs",
                    "default": None,
                    "anyOf": [
                        {
                            "description": "None uses the default classifier, LinearSVC.",
                            "enum": [None],
                        },
                        {"laleType": "operator"},
                    ],
                },
                "method": {
                    "description": "The method to use for calibration. Can be ‘sigmoid’ which corresponds to Platt’s method (i.e. a logistic regression model) or ‘isotonic’ which is a non-parametric approach. It is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit.",
                    "enum": ["sigmoid", "isotonic"],
                    "default": "sigmoid",
                },
                "cv": {
                    "description": "Determines the cross-validation splitting strategy",
                    "default": None,
                    "anyOf": [
                        {
                            "description": "use the default 5-fold cross-validation",
                            "enum": [None],
                        },
                        {
                            "type": "integer",
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 4,
                            "distribution": "uniform",
                        },
                        {
                            "forOptimizer": False,
                            "laleType": "Any",
                            "description": "CV splitter or an iterable yielding (train, test) splits as arrays of indices.",
                        },
                    ],
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
    "description": """`CalibratedClassifierCV`_ : Probability calibration with isotonic regression or sigmoid.

.. _`CalibratedClassifierCV`: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV
    """,
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.calibrated_classifier_cv.html",
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
CalibratedClassifierCV = make_operator(Op, _combined_schemas)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.23/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
    CalibratedClassifierCV = typing.cast(
        lale.operators.PlannedIndividualOp,
        CalibratedClassifierCV.customize_schema(
            n_jobs=AnyOf(
                types=[
                    Int(minimum=1),
                    Int(minimum=-1, maximum=-1, desc="Use all the processors"),
                    Null(),
                ],
                desc="Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.",
                default=None,
            ),
            ensemble=Bool(
                default=True, desc="Determines how the calibrator is fitted."
            ),
        ),
    )

set_docstrings(CalibratedClassifierCV)
