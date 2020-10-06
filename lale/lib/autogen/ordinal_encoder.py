from numpy import inf, nan
from sklearn.preprocessing import OrdinalEncoder as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class OrdinalEncoderImpl:
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
    "description": "inherited docstring for OrdinalEncoder    Encode categorical features as an integer array.",
    "allOf": [
        {
            "type": "object",
            "required": ["categories", "dtype"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "categories": {
                    "XXX TODO XXX": "'auto' or a list of lists/arrays of values.",
                    "description": "Categories (unique values) per feature:  - 'auto' : Determine categories automatically from the training data",
                    "enum": ["auto"],
                    "default": "auto",
                },
                "dtype": {
                    "laleType": "Any",
                    "XXX TODO XXX": "number type, default np.float64",
                    "description": "Desired dtype of output.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the OrdinalEncoder to X.",
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
    "description": "Transform X to ordinal codes.",
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
    "XXX TODO XXX": "sparse matrix or a 2-d array",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OrdinalEncoder#sklearn-preprocessing-ordinalencoder",
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
set_docstrings(OrdinalEncoderImpl, _combined_schemas)
OrdinalEncoder = make_operator(OrdinalEncoderImpl, _combined_schemas)
