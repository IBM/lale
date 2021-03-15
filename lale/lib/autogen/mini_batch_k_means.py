from numpy import inf, nan
from sklearn.cluster import MiniBatchKMeans as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _MiniBatchKMeansImpl:
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

    def predict(self, X):
        return self._wrapped_model.predict(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for MiniBatchKMeans    Mini-Batch K-Means clustering",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_clusters",
                "init",
                "max_iter",
                "batch_size",
                "verbose",
                "compute_labels",
                "random_state",
                "tol",
                "max_no_improvement",
                "init_size",
                "n_init",
                "reassignment_ratio",
            ],
            "relevantToOptimizer": [
                "n_clusters",
                "init",
                "max_iter",
                "batch_size",
                "compute_labels",
                "tol",
                "max_no_improvement",
                "n_init",
            ],
            "additionalProperties": False,
            "properties": {
                "n_clusters": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 8,
                    "distribution": "uniform",
                    "default": 8,
                    "description": "The number of clusters to form as well as the number of centroids to generate.",
                },
                "init": {
                    "enum": ["k-means++", "random", "ndarray"],
                    "default": "k-means++",
                    "description": "Method for initialization, defaults to 'k-means++':  'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
                },
                "batch_size": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 128,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "Size of the mini batches.",
                },
                "verbose": {
                    "anyOf": [{"type": "boolean"}, {"type": "integer"}],
                    "default": 0,
                    "description": "Verbosity mode.",
                },
                "compute_labels": {
                    "type": "boolean",
                    "default": True,
                    "description": "Compute label assignment and inertia for the complete dataset once the minibatch optimization has converged in fit.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Determines random number generation for centroid initialization and random reassignment",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0,
                    "description": "Control early stopping based on the relative center changes as measured by a smoothed, variance-normalized of the mean center squared position changes",
                },
                "max_no_improvement": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 11,
                    "distribution": "uniform",
                    "default": 10,
                    "description": "Control early stopping based on the consecutive number of mini batches that does not yield an improvement on the smoothed inertia",
                },
                "init_size": {
                    "XXX TODO XXX": "int, optional, default: 3 * batch_size",
                    "description": "Number of samples to randomly sample for speeding up the initialization (sometimes at the expense of accuracy): the only algorithm is initialized by running a batch KMeans on a random subset of the data",
                    "enum": [None],
                    "default": None,
                },
                "n_init": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 10,
                    "distribution": "uniform",
                    "default": 3,
                    "description": "Number of random initializations that are tried",
                },
                "reassignment_ratio": {
                    "type": "number",
                    "default": 0.01,
                    "description": "Control the fraction of the maximum number of counts for a center to be reassigned",
                },
            },
        },
        {
            "XXX TODO XXX": "Parameter: init_size > only algorithm is initialized by running a batch kmeans on a random subset of the data"
        },
        {
            "XXX TODO XXX": "Parameter: n_init > only run once, using the best of the n_init initializations as measured by inertia"
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the centroids on X by chunking it into mini-batches.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array-like or sparse matrix, shape=(n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Training instances to cluster",
        },
        "y": {
            "description": "not used, present here for API consistency by convention."
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "The weights for each observation in X",
        },
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X to a cluster-distance space.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "New data to transform.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "X transformed in the new space.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict the closest cluster each sample in X belongs to.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "New data to predict.",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "The weights for each observation in X",
        },
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Index of the cluster each sample belongs to.",
    "type": "array",
    "items": {"type": "number"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.cluster.MiniBatchKMeans#sklearn-cluster-minibatchkmeans",
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
MiniBatchKMeans = make_operator(_MiniBatchKMeansImpl, _combined_schemas)

set_docstrings(MiniBatchKMeans)
