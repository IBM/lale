from numpy import inf, nan
from sklearn.impute import MissingIndicator as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class MissingIndicatorImpl:
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
    "description": "inherited docstring for MissingIndicator    Binary indicators for missing values.",
    "allOf": [
        {
            "type": "object",
            "required": ["missing_values", "features", "sparse", "error_on_new"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "missing_values": {
                    "XXX TODO XXX": "number, string, np.nan (default) or None",
                    "description": "The placeholder for the missing values",
                    "type": "number",
                    "default": nan,
                },
                "features": {
                    "type": "string",
                    "default": "missing-only",
                    "description": "Whether the imputer mask should represent all or a subset of features",
                },
                "sparse": {
                    "XXX TODO XXX": 'boolean or "auto", optional',
                    "description": "Whether the imputer mask format should be sparse or dense",
                    "enum": ["auto"],
                    "default": "auto",
                },
                "error_on_new": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True (default), transform will raise an error when there are features with missing values in transform that have no missing values in fit",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: features > only represent features containing missing values during fit time"
        },
        {
            "description": 'error_on_new, only when features="missing-only"',
            "anyOf": [
                {"type": "object", "properties": {"error_on_new": {"enum": [True]}}},
                {
                    "type": "object",
                    "properties": {"features": {"enum": ["missing-only"]}},
                },
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the transformer on X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Generate missing values indicator for X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The input data to complete.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The missing indicator for input data",
    "laleType": "Any",
    "XXX TODO XXX": "{ndarray or sparse matrix}, shape (n_samples, n_features)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.impute.MissingIndicator#sklearn-impute-missingindicator",
    "import_from": "sklearn.impute",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
set_docstrings(MissingIndicatorImpl, _combined_schemas)
MissingIndicator = make_operator(MissingIndicatorImpl, _combined_schemas)
