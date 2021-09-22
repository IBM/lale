from numpy import inf, nan
from sklearn.linear_model import OrthogonalMatchingPursuitCV as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _OrthogonalMatchingPursuitCVImpl:
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
    "description": "inherited docstring for OrthogonalMatchingPursuitCV    Cross-validated Orthogonal Matching Pursuit model (OMP).",
    "allOf": [
        {
            "type": "object",
            "required": [
                "copy",
                "fit_intercept",
                "normalize",
                "max_iter",
                "cv",
                "n_jobs",
                "verbose",
            ],
            "relevantToOptimizer": [
                "copy",
                "fit_intercept",
                "normalize",
                "max_iter",
                "cv",
            ],
            "additionalProperties": False,
            "properties": {
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the design matrix X must be copied by the algorithm",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "whether to calculate the intercept for this model",
                },
                "normalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "This parameter is ignored when ``fit_intercept`` is set to False",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 10,
                            "maximumForOptimizer": 1000,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Maximum numbers of iterations to perform, therefore maximum features to include",
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
                    ],
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "Number of CPUs to use during the cross validation",
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": False,
                    "description": "Sets the verbosity amount",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: copy > only helpful if x is already fortran-ordered, otherwise a copy is made anyway"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model using X, y as training data.",
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
            "description": "Target values",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict using the linear model",
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
    "description": "Returns predicted values.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV#sklearn-linear_model-orthogonalmatchingpursuitcv",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
OrthogonalMatchingPursuitCV = make_operator(
    _OrthogonalMatchingPursuitCVImpl, _combined_schemas
)

set_docstrings(OrthogonalMatchingPursuitCV)
