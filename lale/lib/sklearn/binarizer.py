from sklearn.preprocessing import Binarizer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Binarize data (set feature values to 0 or 1) according to a threshold",
    "allOf": [
        {
            "type": "object",
            "required": ["threshold", "copy"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "threshold": {
                    "description": "Feature values below or equal to this are replaced by 0, above it by 1. Threshold may not be less than 0 for operations on sparse matrices.",
                    "type": "number",
                    "default": 0.0,
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "set to False to perform inplace binarization and avoid a copy (if the input is already a numpy array or a scipy.sparse CSR matrix).",
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
    "description": "Binarize each element of X",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The data to binarize, element by element",
        },
        "y": {"laleType": "Any", "XXX TODO XXX": "(ignored)", "description": ""},
        "copy": {"type": "boolean", "description": "Copy the input X or not."},
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Binarize each element of X",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Binarize data`_ (set feature values to 0 or 1) according to a threshold

.. _`Binarize data`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.binarizer.html",
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
Binarizer = make_operator(Op, _combined_schemas)

set_docstrings(Binarizer)
