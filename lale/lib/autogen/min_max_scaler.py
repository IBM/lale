from numpy import inf, nan
from sklearn.preprocessing import MinMaxScaler as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _MinMaxScalerImpl:
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
    "description": "inherited docstring for MinMaxScaler    Transforms features by scaling each feature to a given range.",
    "allOf": [
        {
            "type": "object",
            "required": ["feature_range", "copy"],
            "relevantToOptimizer": ["copy"],
            "additionalProperties": False,
            "properties": {
                "feature_range": {
                    "XXX TODO XXX": "tuple (min, max), default=(0, 1)",
                    "description": "Desired range of transformed data.",
                    "type": "array",
                    "laleType": "tuple",
                    "default": (0, 1),
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the minimum and maximum to be used for later scaling.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data used to compute the per-feature minimum and maximum used for later scaling along the features axis.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Scaling features of X according to feature_range.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data that will be transformed.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Scaling features of X according to feature_range.",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.MinMaxScaler#sklearn-preprocessing-minmaxscaler",
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
MinMaxScaler = make_operator(_MinMaxScalerImpl, _combined_schemas)

set_docstrings(MinMaxScaler)
