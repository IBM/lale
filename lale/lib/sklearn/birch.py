from sklearn.cluster import Birch as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Implements the Birch clustering algorithm.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "threshold",
                "branching_factor",
                "n_clusters",
                "compute_labels",
                "copy",
            ],
            "relevantToOptimizer": [
                "branching_factor",
                "n_clusters",
                "compute_labels",
            ],
            "additionalProperties": False,
            "properties": {
                "threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold",
                },
                "branching_factor": {
                    "type": "integer",
                    "minimumForOptimizer": 50,
                    "maximumForOptimizer": 51,
                    "distribution": "uniform",
                    "default": 50,
                    "description": "Maximum number of CF subclusters in each node",
                },
                "n_clusters": {
                    "description": "Number of clusters after the final clustering step, which treats the subclusters from the leaves as new samples",
                    "default": 3,
                    "anyOf": [
                        {
                            "description": "The model fit is AgglomerativeClustering with n_clusters set to be equal to the int.",
                            "type": "integer",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 8,
                            "distribution": "uniform",
                        },
                        {
                            "forOptimizer": False,
                            "description": "sklearn.cluster Estimator: If a model is provided, the model is fit treating the subclusters as new samples and the initial data is mapped to the label of the closest subcluster.",
                            "laleType": "operator",
                        },
                        {
                            "enum": [None],
                            "description": "The final clustering step is not performed and the subclusters are returned as they are.",
                        },
                    ],
                },
                "compute_labels": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not to compute labels for each fit.",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not to make a copy of the given data",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Build a CF Tree for the input data.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X into subcluster centroids dimension.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict data using the ``centroids_`` of subclusters.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data.",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Labelled data.",
    "laleType": "Any",
    "XXX TODO XXX": "ndarray, shape(n_samples)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Birch`_ clustering algorithm.

    .. _`Birch`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch
    """,
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.birch.html",
    "import_from": "sklearn.cluster",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
Birch = make_operator(Op, _combined_schemas)

set_docstrings(Birch)
