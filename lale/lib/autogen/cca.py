from numpy import inf, nan
from sklearn.cross_decomposition import CCA as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _CCAImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)

    def predict(self, X):
        return self._wrapped_model.predict(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for CCA    CCA Canonical Correlation Analysis.",
    "allOf": [
        {
            "type": "object",
            "required": ["n_components", "scale", "max_iter", "tol", "copy"],
            "relevantToOptimizer": ["n_components", "scale", "max_iter", "tol", "copy"],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 256,
                    "distribution": "uniform",
                    "default": 2,
                    "description": "number of components to keep.",
                },
                "scale": {
                    "type": "boolean",
                    "default": True,
                    "description": "whether to scale the data?",
                },
                "max_iter": {
                    "XXX TODO XXX": "an integer, (default 500)",
                    "description": "the maximum number of iterations of the NIPALS inner loop",
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 500,
                },
                "tol": {
                    "XXX TODO XXX": "non-negative real, default 1e-06.",
                    "description": "the tolerance used in the iterative algorithm",
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 1e-06,
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the deflation be done on a copy",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit model to data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of predictors.",
        },
        "Y": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.",
        },
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply the dimension reduction learned on the train data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of predictors.",
        },
        "Y": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.",
        },
        "copy": {
            "type": "boolean",
            "default": True,
            "description": "Whether to copy X and Y, or perform in-place normalization.",
        },
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply the dimension reduction learned on the train data.",
    "laleType": "Any",
    "XXX TODO XXX": "x_scores if Y is not given, (x_scores, y_scores) otherwise.",
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply the dimension reduction learned on the train data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of predictors.",
        },
        "copy": {
            "type": "boolean",
            "default": True,
            "description": "Whether to copy X and Y, or perform in-place normalization.",
        },
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply the dimension reduction learned on the train data.",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.cross_decomposition.CCA#sklearn-cross_decomposition-cca",
    "import_from": "sklearn.cross_decomposition",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
CCA = make_operator(_CCAImpl, _combined_schemas)

set_docstrings(CCA)
