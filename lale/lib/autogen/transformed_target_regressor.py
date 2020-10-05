from numpy import inf, nan
from sklearn.compose import TransformedTargetRegressor as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class TransformedTargetRegressorImpl:
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
    "description": "inherited docstring for TransformedTargetRegressor    Meta-estimator to regress on a transformed target.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "regressor",
                "transformer",
                "func",
                "inverse_func",
                "check_inverse",
            ],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "regressor": {
                    "XXX TODO XXX": "object, default=LinearRegression()",
                    "description": "Regressor object such as derived from ``RegressorMixin``",
                    "enum": [None],
                    "default": None,
                },
                "transformer": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Estimator object such as derived from ``TransformerMixin``",
                },
                "func": {
                    "XXX TODO XXX": "function, optional",
                    "description": "Function to apply to ``y`` before passing to ``fit``",
                    "enum": [None],
                    "default": None,
                },
                "inverse_func": {
                    "XXX TODO XXX": "function, optional",
                    "description": "Function to apply to the prediction of the regressor",
                    "enum": [None],
                    "default": None,
                },
                "check_inverse": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to check that ``transform`` followed by ``inverse_transform`` or ``func`` followed by ``inverse_func`` leads to the original targets.",
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
            "description": "Training vector, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values.",
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
    "description": "Predict using the base regressor, applying inverse.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Samples.",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predicted values.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.compose.TransformedTargetRegressor#sklearn-compose-transformedtargetregressor",
    "import_from": "sklearn.compose",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
set_docstrings(TransformedTargetRegressorImpl, _combined_schemas)
TransformedTargetRegressor = make_operator(
    TransformedTargetRegressorImpl, _combined_schemas
)
