from numpy import inf, nan
from sklearn.preprocessing import OneHotEncoder as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class OneHotEncoderImpl:
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
    "description": "inherited docstring for OneHotEncoder    Encode categorical integer features as a one-hot numeric array.",
    "allOf": [
        {
            "type": "object",
            "required": ["categories", "drop", "sparse", "dtype", "handle_unknown",],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "categories": {
                    "XXX TODO XXX": "'auto' or a list of lists/arrays of values, default='auto'.",
                    "description": "Categories (unique values) per feature:  - 'auto' : Determine categories automatically from the training data",
                    "enum": ["auto"],
                    "default": "auto",
                },
                "drop": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"enum": ["first", "if_binary", None]},
                    ],
                    "default": None,
                    "description": "Specifies a methodology to use to drop one of the categories per feature.",
                },
                "sparse": {
                    "type": "boolean",
                    "default": True,
                    "description": "Will return sparse matrix if set True else will return an array.",
                },
                "dtype": {
                    "laleType": "Any",
                    "XXX TODO XXX": "number type, default=np.float",
                    "description": "Desired dtype of output.",
                },
                "handle_unknown": {
                    "XXX TODO XXX": "'error' or 'ignore', default='error'.",
                    "description": "Whether to raise an error or ignore if an unknown categorical feature is present during transform (default is to raise)",
                    "enum": ["error"],
                    "default": "error",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit OneHotEncoder to X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data to determine the categories of each feature.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X using one-hot encoding.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data to encode.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed input.",
    "laleType": "Any",
    "XXX TODO XXX": "sparse matrix if sparse=True else a 2-d array",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OneHotEncoder#sklearn-preprocessing-onehotencoder",
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
set_docstrings(OneHotEncoderImpl, _combined_schemas)
OneHotEncoder = make_operator(OneHotEncoderImpl, _combined_schemas)
