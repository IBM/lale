from numpy import inf, nan
from sklearn.neighbors import KNeighborsClassifier as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class KNeighborsClassifierImpl:
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

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for KNeighborsClassifier    Classifier implementing the k-nearest neighbors vote.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_neighbors",
                "weights",
                "algorithm",
                "leaf_size",
                "p",
                "metric",
                "metric_params",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "n_neighbors",
                "weights",
                "algorithm",
                "leaf_size",
                "p",
                "metric",
            ],
            "additionalProperties": False,
            "properties": {
                "n_neighbors": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 20,
                    "distribution": "uniform",
                    "default": 5,
                    "description": "Number of neighbors to use by default for :meth:`kneighbors` queries.",
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
                    "minimumForOptimizer": 2,
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
                "metric_params": {
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                    "default": None,
                    "description": "Additional keyword arguments for the metric function.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
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
            "laleType": "Any",
            "XXX TODO XXX": "array-like, shape (n_query, n_features),                 or (n_query, n_indexed) if metric == 'precomputed'",
            "description": "Test samples.",
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
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Return probability estimates for the test data X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "array-like, shape (n_query, n_features),                 or (n_query, n_indexed) if metric == 'precomputed'",
            "description": "Test samples.",
        }
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "of such arrays if n_outputs > 1",
    "laleType": "Any",
    "XXX TODO XXX": "array of shape = [n_samples, n_classes], or a list of n_outputs",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.neighbors.KNeighborsClassifier#sklearn-neighbors-kneighborsclassifier",
    "import_from": "sklearn.neighbors",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}
set_docstrings(KNeighborsClassifierImpl, _combined_schemas)
KNeighborsClassifier = make_operator(KNeighborsClassifierImpl, _combined_schemas)
