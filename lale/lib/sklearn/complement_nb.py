from sklearn.naive_bayes import ComplementNB as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "The Complement Naive Bayes classifier described in Rennie et al. (2003).",
    "allOf": [
        {
            "type": "object",
            "required": ["alpha", "fit_prior", "class_prior", "norm"],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "alpha": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).",
                },
                "fit_prior": {
                    "type": "boolean",
                    "default": True,
                    "description": "Only used in edge case with a single class in the training set.",
                },
                "class_prior": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Prior probabilities of the classes.  Not used.",
                },
                "norm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether or not a second normalization of the weights is performed",
                },
            },
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit Naive Bayes classifier according to X, y",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values.",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Weights applied to individual samples (1",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Perform classification on an array of test vectors X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predicted target values for X",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Return probability estimates for the test vector X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns the probability of the samples for each class in the model",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Complement Naive Bayes`_ classifier described in Rennie et al. (2003).

.. _`Complement Naive Bayes`: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB
    """,
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.complement_nb.html",
    "import_from": "sklearn.naive_bayes",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}
ComplementNB = make_operator(Op, _combined_schemas)

set_docstrings(ComplementNB)
