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

import numpy as np
import sklearn.cluster

import lale.docstrings
import lale.operators


class FeatureAgglomerationImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.cluster.FeatureAgglomeration(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "description": "Agglomerate features.",
    "allOf": [
        {
            "type": "object",
            "required": ["memory", "compute_full_tree", "pooling_func"],
            "relevantToOptimizer": ["affinity", "compute_full_tree", "linkage"],
            "additionalProperties": False,
            "properties": {
                "n_clusters": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 8,
                    "default": 2,
                    "laleMaximum": "X/maxItems",  # number of rows
                    "description": "The number of clusters to find.",
                },
                "affinity": {
                    "anyOf": [
                        {
                            "enum": [
                                "euclidean",
                                "l1",
                                "l2",
                                "manhattan",
                                "cosine",
                                "precomputed",
                            ]
                        },
                        {"forOptimizer": False, "laleType": "callable"},
                    ],
                    "default": "euclidean",
                    "description": "Metric used to compute the linkage.",
                },
                "memory": {
                    "anyOf": [
                        {
                            "description": "Path to the caching directory.",
                            "type": "string",
                        },
                        {
                            "description": "Object with the joblib.Memory interface",
                            "type": "object",
                            "forOptimizer": False,
                        },
                        {"description": "No caching.", "enum": [None]},
                    ],
                    "default": None,
                    "description": "Used to cache the output of the computation of the tree.",
                },
                "connectivity": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        {
                            "laleType": "callable",
                            "forOptimizer": False,
                            "description": "A callable that transforms the data into a connectivity matrix, such as derived from kneighbors_graph.",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Connectivity matrix. Defines for each feature the neighboring features following a given structure of the data.",
                },
                "compute_full_tree": {
                    "anyOf": [{"type": "boolean"}, {"enum": ["auto"]}],
                    "default": "auto",
                    "description": "Stop early the construction of the tree at n_clusters.",
                },
                "linkage": {
                    "enum": ["ward", "complete", "average", "single"],
                    "default": "ward",
                    "description": "Which linkage criterion to use. The linkage criterion determines which distance to use between sets of features.",
                },
                "pooling_func": {
                    "description": "This combines the values of agglomerated features into a single value, and should accept an array of shape [M, N] and the keyword argument axis=1, and reduce it to an array of size [M].",
                    "laleType": "callable",
                    "default": np.mean,
                },
            },
        },
        {
            "description": 'affinity, if linkage is "ward", only "euclidean" is accepted',
            "anyOf": [
                {"type": "object", "properties": {"affinity": {"enum": ["euclidean"]}}},
                {
                    "type": "object",
                    "properties": {"linkage": {"not": {"enum": ["ward"]}}},
                },
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Fit the hierarchical clustering on the data",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "The data",
        },
        "y": {"description": "Ignored"},
    },
}

_input_transform_schema = {
    "description": "Transform a new matrix using the built clustering",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"},},
            "description": "A M by N array of M observations in N dimensions or a length",
        },
    },
}
_output_transform_schema = {
    "description": "The pooled values for each feature cluster.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"},},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Feature agglomeration`_ transformer from scikit-learn.

.. _`Feature agglomeration`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.feature_agglomeration.html",
    "import_from": "sklearn.cluster",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

FeatureAgglomeration: lale.operators.IndividualOp
FeatureAgglomeration = lale.operators.make_operator(
    FeatureAgglomerationImpl, _combined_schemas
)

if sklearn.__version__ >= "0.21":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.cluster.FeatureAgglomeration.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.cluster.FeatureAgglomeration.html
    from lale.schemas import AnyOf, Enum, Float, Int, Null, Object

    FeatureAgglomeration = FeatureAgglomeration.customize_schema(
        distance_threshold=AnyOf(
            types=[Float(), Null()],
            desc="The linkage distance threshold above which, clusters will not be merged.",
            default=None,
        ),
        n_clusters=AnyOf(
            types=[
                Int(minForOptimizer=2, maxForOptimizer=8, laleMaximum="X/maxItems"),
                Null(forOptimizer=False),
            ],
            default=2,
            forOptimizer=False,
            desc="The number of clusters to find.",
        ),
        constraint=AnyOf(
            [Object(n_clusters=Null()), Object(distance_threshold=Null())],
            desc="n_clusters must be None if distance_threshold is not None.",
        ),
    )
    FeatureAgglomeration = FeatureAgglomeration.customize_schema(
        constraint=AnyOf(
            [
                Object(compute_full_tree=Enum(["True"])),
                Object(distance_threshold=Null()),
            ],
            desc="compute_full_tree must be True if distance_threshold is not None.",
        )
    )

lale.docstrings.set_docstrings(FeatureAgglomerationImpl, FeatureAgglomeration._schemas)
