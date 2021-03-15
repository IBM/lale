from numpy import inf, nan
from sklearn.preprocessing import LabelBinarizer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _LabelBinarizerImpl:
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
    "description": "inherited docstring for LabelBinarizer    Binarize labels in a one-vs-all fashion",
    "allOf": [
        {
            "type": "object",
            "required": ["neg_label", "pos_label", "sparse_output"],
            "relevantToOptimizer": ["neg_label", "pos_label", "sparse_output"],
            "additionalProperties": False,
            "properties": {
                "neg_label": {
                    "type": "integer",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                    "distribution": "uniform",
                    "default": 0,
                    "description": "Value with which negative labels must be encoded.",
                },
                "pos_label": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 2,
                    "distribution": "uniform",
                    "default": 1,
                    "description": "Value with which positive labels must be encoded.",
                },
                "sparse_output": {
                    "type": "boolean",
                    "default": False,
                    "description": "True if the returned array from transform is desired to be in sparse CSR format.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit label binarizer",
    "type": "object",
    "required": ["y"],
    "properties": {
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Target values",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform multi-class labels to binary labels",
    "type": "object",
    "required": ["y"],
    "properties": {
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]",
                },
                {"type": "array", "items": {"type": "number"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Target values",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Shape will be [n_samples, 1] for binary problems.",
    "laleType": "Any",
    "XXX TODO XXX": "numpy array or CSR matrix of shape [n_samples, n_classes]",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.LabelBinarizer#sklearn-preprocessing-labelbinarizer",
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
LabelBinarizer = make_operator(_LabelBinarizerImpl, _combined_schemas)

set_docstrings(LabelBinarizer)
