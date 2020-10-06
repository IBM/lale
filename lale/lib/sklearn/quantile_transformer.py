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

from sklearn.preprocessing import QuantileTransformer as SKLModel

import lale.docstrings
import lale.operators


class QuantileTransformerImpl:
    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=100000,
        random_state=None,
        copy=True,
    ):
        self._hyperparams = {
            "n_quantiles": n_quantiles,
            "output_distribution": output_distribution,
            "ignore_implicit_zeros": ignore_implicit_zeros,
            "subsample": subsample,
            "random_state": random_state,
            "copy": copy,
        }
        self._wrapped_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "description": "inherited docstring for QuantileTransformer    Transform features using quantiles information.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_quantiles",
                "output_distribution",
                "ignore_implicit_zeros",
                "subsample",
                "random_state",
                "copy",
            ],
            "relevantToOptimizer": ["n_quantiles", "output_distribution", "subsample"],
            "additionalProperties": False,
            "properties": {
                "n_quantiles": {
                    "type": "integer",
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 2000,
                    "distribution": "uniform",
                    "default": 1000,
                    "description": "Number of quantiles to be computed. It corresponds to the number",
                },
                "output_distribution": {
                    "enum": ["normal", "uniform"],
                    "default": "uniform",
                    "description": "Marginal distribution for the transformed data. The choices are",
                },
                "ignore_implicit_zeros": {
                    "type": "boolean",
                    "default": False,
                    "description": "Only applies to sparse matrices. If True, the sparse entries of the",
                },
                "subsample": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 100000,
                    "distribution": "uniform",
                    "default": 100000,
                    "description": "Maximum number of samples used to estimate the quantiles for",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator;",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Set to False to perform inplace transformation and avoid a copy (if the",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Compute the quantiles used for transforming.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "description": "The data used to scale along the features axis. If a sparse matrix is provided, "
            "it will be converted into a sparse csc_matrix. Additionally, "
            "the sparse matrix needs to be nonnegative if ignore_implicit_zeros is False.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}
_input_transform_schema = {
    "description": "Feature-wise transformation of the data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "description": "The data used to scale along the features axis. If a sparse matrix is provided, "
            "it will be converted into a sparse csc_matrix. Additionally, "
            "the sparse matrix needs to be nonnegative if ignore_implicit_zeros is False.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}
_output_transform_schema = {
    "description": "The projected data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Quantile transformer`_ from scikit-learn.

.. _`Quantile transformer`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.quantile_transformer.html",
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

lale.docstrings.set_docstrings(QuantileTransformerImpl, _combined_schemas)

QuantileTransformer = lale.operators.make_operator(
    QuantileTransformerImpl, _combined_schemas
)
