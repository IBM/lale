from numpy import inf, nan
from sklearn.preprocessing import KBinsDiscretizer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _KBinsDiscretizerImpl:
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
    "description": "inherited docstring for KBinsDiscretizer    Bin continuous data into intervals.",
    "allOf": [
        {
            "type": "object",
            "required": ["n_bins", "encode", "strategy"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "n_bins": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "array", "items": {"type": "number"}},
                    ],
                    "default": 5,
                    "description": "The number of bins to produce",
                },
                "encode": {
                    "enum": ["onehot", "onehot-dense", "ordinal"],
                    "default": "onehot",
                    "description": "Method used to encode the transformed result",
                },
                "strategy": {
                    "enum": ["uniform", "quantile", "kmeans"],
                    "default": "quantile",
                    "description": "Strategy used to define the widths of the bins",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fits the estimator.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "numeric array-like, shape (n_samples, n_features)",
            "description": "Data to be discretized.",
        },
        "y": {"laleType": "Any", "XXX TODO XXX": "ignored"},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Discretizes the data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "numeric array-like, shape (n_samples, n_features)",
            "description": "Data to be discretized.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Data in the binned space.",
    "laleType": "Any",
    "XXX TODO XXX": "numeric array-like or sparse matrix",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.KBinsDiscretizer#sklearn-preprocessing-kbinsdiscretizer",
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
KBinsDiscretizer = make_operator(_KBinsDiscretizerImpl, _combined_schemas)

set_docstrings(KBinsDiscretizer)
