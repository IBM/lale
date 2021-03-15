# Copyright 2020 IBM Corporation
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

import autoai_libs.transformers.text_transformers

import lale.docstrings
import lale.operators


# This is currently needed just to hide get_params so that lale does not call clone
# when doing a defensive copy
class _Word2VecTransformerImpl:
    def __init__(
        self,
        output_dim=30,
        column_headers_list=[],
        svd_num_iter=5,
        drop_columns=False,
        activate_flag=True,
        min_count=5,
        text_columns=None,
        text_processing_options={},
    ):
        self._hyperparams = {
            "output_dim": output_dim,
            "column_headers_list": column_headers_list,
            "svd_num_iter": svd_num_iter,
            "drop_columns": drop_columns,
            "activate_flag": activate_flag,
            "min_count": min_count,
            "text_columns": text_columns,
            "text_processing_options": text_processing_options,
        }
        self._wrapped_model = (
            autoai_libs.transformers.text_transformers.Word2VecTransformer(
                **self._hyperparams
            )
        )

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "allOf": [
        {
            "description": """This transformer converts text columns in the dataset to its word2vec embedding vectors.
It then performs SVD on those vectors for dimensionality reduction.""",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "output_dim",
                "column_headers_list",
                "svd_num_iter",
                "drop_columns",
                "activate_flag",
                "min_count",
                "text_columns",
                "text_processing_options",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "output_dim": {
                    "description": "Number of numeric features generated per text column.",
                    "type": "integer",
                    "default": 30,
                },
                "column_headers_list": {
                    "description": """Column headers passed from autoai_core. The new feature's column headers are
appended to this.""",
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "array", "items": {"type": "integer"}},
                    ],
                    "default": [],
                },
                "svd_num_iter": {
                    "description": "Number of iterations for which svd was run.",
                    "type": "integer",
                    "default": 5,
                },
                "drop_columns": {
                    "description": "If true, drops text columns",
                    "type": "boolean",
                    "default": False,
                },
                "activate_flag": {
                    "description": "If False, the features are not generated.",
                    "type": "boolean",
                    "default": True,
                },
                "min_count": {
                    "description": "Word2vec model ignores all the words whose frequency is less than this.",
                    "type": "integer",
                    "default": 5,
                },
                "text_columns": {
                    "description": "If passed, then word2vec features are applied to these columns.",
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "array", "items": {"type": "integer"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "text_processing_options": {
                    "description": "The parameter values to initialize this transformer are passed through this dictionary.",
                    "type": "object",
                    "default": {},
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        },
        "y": {"laleType": "Any"},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Converts text columns to numeric features using a combination of word2vec and SVD.
.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.word2vec_transformer.html",
    "import_from": "autoai_libs.transformers.text_transformers",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

Word2VecTransformer = lale.operators.make_operator(
    _Word2VecTransformerImpl, _combined_schemas
)

lale.docstrings.set_docstrings(Word2VecTransformer)
