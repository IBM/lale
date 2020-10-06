from numpy import inf, nan
from sklearn.preprocessing import Normalizer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class NormalizerImpl:
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
    "description": "inherited docstring for Normalizer    Normalize samples individually to unit norm.",
    "allOf": [
        {
            "type": "object",
            "required": ["norm", "copy"],
            "relevantToOptimizer": ["norm", "copy"],
            "additionalProperties": False,
            "properties": {
                "norm": {
                    "XXX TODO XXX": "'l1', 'l2', or 'max', optional ('l2' by default)",
                    "description": "The norm to use to normalize each non zero sample.",
                    "enum": ["l2"],
                    "default": "l2",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array or a scipy.sparse CSR matrix).",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Do nothing and return the estimator unchanged",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
            "XXX TODO XXX": "array-like",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Scale each non zero row of X to unit norm",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data to normalize, row by row",
        },
        "y": {"laleType": "Any", "XXX TODO XXX": "(ignored)", "description": ""},
        "copy": {
            "anyOf": [{"type": "boolean"}, {"enum": [None]}],
            "default": None,
            "description": "Copy the input X or not.",
        },
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Scale each non zero row of X to unit norm",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.Normalizer#sklearn-preprocessing-normalizer",
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
set_docstrings(NormalizerImpl, _combined_schemas)
Normalizer = make_operator(NormalizerImpl, _combined_schemas)
