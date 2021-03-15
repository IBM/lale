from numpy import inf, nan
from sklearn.preprocessing import FunctionTransformer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _FunctionTransformerImpl:
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


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for FunctionTransformer    Constructs a transformer from an arbitrary callable.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "func",
                "inverse_func",
                "validate",
                "accept_sparse",
                "check_inverse",
                "kw_args",
                "inv_kw_args",
            ],
            "relevantToOptimizer": ["accept_sparse"],
            "additionalProperties": False,
            "properties": {
                "func": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "The callable to use for the transformation",
                },
                "inverse_func": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "The callable to use for the inverse transformation",
                },
                "validate": {
                    "anyOf": [{"type": "boolean"}, {"enum": [None]}],
                    "default": None,
                    "description": "Indicate that the input X array should be checked before calling ``func``",
                },
                "accept_sparse": {
                    "type": "boolean",
                    "default": False,
                    "description": "Indicate that func accepts a sparse matrix as input",
                },
                "check_inverse": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to check that or ``func`` followed by ``inverse_func`` leads to the original inputs",
                },
                "kw_args": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Dictionary of additional keyword arguments to pass to func.",
                },
                "inv_kw_args": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Dictionary of additional keyword arguments to pass to inverse_func.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit transformer by checking X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input array.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X using the forward function.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input array.",
        },
        "y": {"laleType": "Any", "XXX TODO XXX": "(ignored)", "description": ""},
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed input.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.FunctionTransformer#sklearn-preprocessing-functiontransformer",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
FunctionTransformer = make_operator(_FunctionTransformerImpl, _combined_schemas)

set_docstrings(FunctionTransformer)
