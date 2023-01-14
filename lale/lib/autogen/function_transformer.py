from numpy import inf, nan
from packaging import version
from sklearn.preprocessing import FunctionTransformer as Op

import lale
from lale.docstrings import set_docstrings
from lale.operators import make_operator, sklearn_version


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

if sklearn_version >= version.Version("0.22"):
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    from lale.schemas import Bool

    FunctionTransformer = FunctionTransformer.customize_schema(
        validate=Bool(
            desc="Indicate that the input X array should be checked before calling ``func``.",
            default=False,
        ),
        pass_y=None,
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.1"):
    # old: https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    # new: https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.FunctionTransformer.html
    FunctionTransformer = FunctionTransformer.customize_schema(
        feature_names_out={
            "anyOf": [{"laleType": "callable"}, {"enum": ["one-to-one", None]}],
            "default": None,
            "description": "Determines the list of feature names that will be returned by the ``get_feature_names_out`` method. If it is ‘one-to-one’, then the output feature names will be equal to the input feature names. If it is a callable, then it must take two positional arguments: this ``FunctionTransformer`` (``self``) and an array-like of input feature names (``input_features``). It must return an array-like of output feature names. The ``get_feature_names_out`` method is only defined if ``feature_names_out`` is not None.",
        },
        set_as_available=True,
    )
set_docstrings(FunctionTransformer)
