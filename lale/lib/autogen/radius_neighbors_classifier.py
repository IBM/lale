from numpy import inf, nan
from packaging import version
from sklearn.neighbors import RadiusNeighborsClassifier as Op

import lale.operators
from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _RadiusNeighborsClassifierImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for RadiusNeighborsClassifier    Classifier implementing a vote among neighbors within a given radius",
    "allOf": [
        {
            "type": "object",
            "required": [
                "radius",
                "weights",
                "algorithm",
                "leaf_size",
                "p",
                "metric",
                "outlier_label",
                "metric_params",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "weights",
                "algorithm",
                "leaf_size",
                "p",
                "metric",
                "outlier_label",
            ],
            "additionalProperties": False,
            "properties": {
                "radius": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Range of parameter space to use by default for :meth:`radius_neighbors` queries.",
                },
                "weights": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": ["distance", "uniform"]},
                    ],
                    "default": "uniform",
                    "description": "weight function used in prediction",
                },
                "algorithm": {
                    "enum": ["auto", "ball_tree", "kd_tree", "brute"],
                    "default": "auto",
                    "description": "Algorithm used to compute the nearest neighbors:  - 'ball_tree' will use :class:`BallTree` - 'kd_tree' will use :class:`KDTree` - 'brute' will use a brute-force search",
                },
                "leaf_size": {
                    "type": "integer",
                    "minimumForOptimizer": 30,
                    "maximumForOptimizer": 31,
                    "distribution": "uniform",
                    "default": 30,
                    "description": "Leaf size passed to BallTree or KDTree",
                },
                "p": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 3,
                    "distribution": "uniform",
                    "default": 2,
                    "description": "Power parameter for the Minkowski metric",
                },
                "metric": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {
                            "enum": [
                                "euclidean",
                                "manhattan",
                                "minkowski",
                                "precomputed",
                            ]
                        },
                    ],
                    "default": "minkowski",
                    "description": "the distance metric to use for the tree",
                },
                "outlier_label": {
                    "anyOf": [
                        {"type": "integer", "forOptimizer": False},
                        {"enum": ["most_frequent"]},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Label, which is given for outlier samples (samples with no neighbors on given radius)",
                },
                "metric_params": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Additional keyword arguments for the metric function.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of parallel jobs to run for neighbors search",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model using X as training data and y as target values",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
            "XXX TODO XXX": "{array-like, sparse matrix, BallTree, KDTree}",
            "description": "Training data",
        },
        "y": {
            "type": "array",
            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
            "XXX TODO XXX": "{array-like, sparse matrix}",
            "description": "Target values of shape = [n_samples] or [n_samples, n_outputs]",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict the class labels for the provided data",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {
                    "enum": [None],
                    "description": "Predictions for all training set points are returned, and points are not included into their own neighbors",
                },
            ],
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Class labels for each data sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier#sklearn-neighbors-radiusneighborsclassifier",
    "import_from": "sklearn.neighbors",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
RadiusNeighborsClassifier = make_operator(
    _RadiusNeighborsClassifierImpl, _combined_schemas
)

if lale.operators.sklearn_version >= version.Version("1.6"):
    RadiusNeighborsClassifier = RadiusNeighborsClassifier.customize_schema(
        metric={
            "anyOf": [
                {"enum": ["euclidean", "manhattan", "minkowski", "nan_euclidean"]},
                {
                    "laleType": "callable",
                    "forOptimizer": False,
                    "description": "Takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors.",
                },
            ],
            "description": "The distance metric to use for the tree.",
            "default": "minkowski",
        },
        set_as_available=True,
    )

set_docstrings(RadiusNeighborsClassifier)
