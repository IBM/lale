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

from sklearn.manifold import Isomap as SKLModel

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for Isomap    Isomap Embedding",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_neighbors",
                "n_components",
                "eigen_solver",
                "tol",
                "max_iter",
                "path_method",
                "neighbors_algorithm",
                "n_jobs",
                "metric",
                "p",
                "metric_params",
            ],
            "relevantToOptimizer": [
                "n_neighbors",
                "n_components",
                "eigen_solver",
                "tol",
                "path_method",
                "neighbors_algorithm",
            ],
            "additionalProperties": False,
            "properties": {
                "n_neighbors": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 20,
                    "distribution": "uniform",
                    "default": 5,
                    "description": "number of neighbors to consider for each point.",
                },
                "n_components": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 256,
                    "distribution": "uniform",
                    "default": 2,
                    "laleMaximum": "X/items/maxItems",
                    "description": "number of coordinates for the manifold",
                },
                "eigen_solver": {
                    "enum": ["auto", "arpack", "dense"],
                    "default": "auto",
                    "description": "'auto' : Attempt to choose the most efficient solver for the given problem",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                    "distribution": "uniform",
                    "default": 0,
                    "description": "Convergence tolerance passed to arpack or lobpcg",
                },
                "max_iter": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Maximum number of iterations for the arpack solver",
                },
                "path_method": {
                    "enum": ["auto", "FW", "D"],
                    "default": "auto",
                    "description": "Method to use in finding shortest path",
                },
                "neighbors_algorithm": {
                    "enum": ["auto", "brute", "kd_tree", "ball_tree"],
                    "default": "auto",
                    "description": "Algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance.",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of parallel jobs to run",
                },
                "metric": {
                    "description": """The metric to use when calculating distance between instances in a feature array.
If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter.
If metric is “precomputed”, X is assumed to be a distance matrix and must be square.""",
                    "default": "minkowski",
                    "laleType": "Any",
                },
                "p": {
                    "description": """Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances.
When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
For arbitrary p, minkowski_distance (l_p) is used.""",
                    "type": "integer",
                    "default": 2,
                },
                "metric_params": {
                    "description": "Additional keyword arguments for the metric function",
                    "default": None,
                    "anyOf": [{"type": "object"}, {"enum": [None]}],
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Compute the embedding vectors for data X",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "laleType": "Any",
            "description": "Sample data, shape = (n_samples, n_features), in the form of a numpy array, precomputed tree, or NearestNeighbors object.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X.",
    "type": "object",
    "required": ["X"],
    "properties": {"X": {"laleType": "Any"}},
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transform X.",
    "laleType": "Any",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """"`Isomap`_ embedding from scikit-learn.

.. _`Isomap`: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.isomap.html",
    "import_from": "sklearn.manifold",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
Isomap = make_operator(SKLModel, _combined_schemas)

set_docstrings(Isomap)
