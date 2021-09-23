# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sklearn
from sklearn.cluster import KMeans as SKLModel

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """The k-means problem is solved using either Lloyd's or Elkan's algorithm.
The average complexity is given by O(k n T), where n is the number of
samples and T is the number of iteration.
The worst case complexity is given by O(n^(k+2/p)) with
n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
'How slow is the k-means method?' SoCG2006)
In practice, the k-means algorithm is very fast (one of the fastest
clustering algorithms available), but it falls in local minima. That's why
it can be useful to restart it several times.
If the algorithm stops before fully converging (because of ``tol`` or
``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
i.e. the ``cluster_centers_`` will not be the means of the points in each
cluster. Also, the estimator will reassign ``labels_`` after the last
iteration to make ``labels_`` consistent with ``predict`` on the training
set.""",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_clusters",
                "init",
                "n_init",
                "max_iter",
                "tol",
                "precompute_distances",
                "verbose",
                "random_state",
                "copy_x",
                "n_jobs",
                "algorithm",
            ],
            "relevantToOptimizer": [
                "n_clusters",
                "init",
                "n_init",
                "max_iter",
                "tol",
                "precompute_distances",
                "copy_x",
                "algorithm",
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
                    "anyOf": [
                        {"enum": ["k-means++", "random"]},
                        {"laleType": "callable", "forOptimizer": False},
                        {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "forOptimizer": False,
                        },
                    ],
                    "default": "k-means++",
                    "description": """Method for initialization, defaults to `k-means++`.
`k-means++` : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
See section Notes in k_init for more details.
`random`: choose n_clusters observations (rows) at random from data for the initial centroids.
If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.""",
                },
                "n_init": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 10,
                    "distribution": "uniform",
                    "default": 10,
                    "description": """Number of time the k-means algorithm will be run with different centroid seeds.
The final results will be the best output of n_init consecutive runs in terms of inertia.""",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 300,
                    "description": "Maximum number of iterations of the k-means algorithm for a single run.",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.",
                },
                "precompute_distances": {
                    "enum": ["auto", True, False],
                    "default": "auto",
                    "description": "Precompute distances (faster but takes more memory). Deprecated.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Verbosity mode.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Determines random number generation for centroid initialization",
                },
                "copy_x": {
                    "type": "boolean",
                    "default": True,
                    "description": """When pre-computing distances it is more numerically accurate to center the data first.
If copy_x is True (default), then the original data is not modified.
If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean.
Note that if the original data is not C-contiguous, a copy will be made even if copy_x is False.
If the original data is sparse, but not in CSR format, a copy will be made even if copy_x is False.""",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "The number of jobs to use for the computation. Deprecated.",
                },
                "algorithm": {
                    "description": """K-means algorithm to use.
The classical EM-style algorithm is “full”. The “elkan” variation is more efficient on data with well-defined clusters, by using the triangle inequality.
However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
For now “auto” (kept for backward compatibiliy) chooses “elkan” but it might change in the future for a better heuristic.""",
                    "enum": ["auto", "full", "elkan"],
                    "default": "auto",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute k-means clustering.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training instances to cluster. Array-like or sparse matrix, shape=(n_samples, n_features)",
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
    "description": """`KMeans`_ from scikit-learn.

.. _`KMeans`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.k_means.html",
    "import_from": "sklearn.cluster",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "clustering", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}
KMeans = make_operator(SKLModel, _combined_schemas)

if sklearn.__version__ >= "1.0":
    # old: https://scikit-learn.org/0.24/modules/generated/sklearn.cluster.KMeans.html
    # new: https://scikit-learn.org/1.0/modules/generated/sklearn.cluster.KMeans.html
    KMeans = KMeans.customize_schema(
        precompute_distances=None,
        n_jobs=None,
        set_as_available=True,
    )

set_docstrings(KMeans)
