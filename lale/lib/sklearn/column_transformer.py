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
import sklearn.compose

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["transformers"],
            "relevantToOptimizer": [],
            "properties": {
                "transformers": {
                    "description": "Operators or pipelines to be applied to subsets of the data.",
                    "type": "array",
                    "items": {
                        "description": "Tuple of (name, transformer, column(s)).",
                        "type": "array",
                        "laleType": "tuple",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": [
                            {"description": "Name.", "type": "string"},
                            {
                                "description": "Transformer.",
                                "anyOf": [
                                    {
                                        "description": "Transformer supporting fit and transform.",
                                        "laleType": "operator",
                                    },
                                    {"enum": ["passthrough", "drop"]},
                                ],
                            },
                            {
                                "description": "Column(s).",
                                "anyOf": [
                                    {
                                        "type": "integer",
                                        "description": "One column by index.",
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "Multiple columns by index.",
                                    },
                                    {
                                        "type": "string",
                                        "description": "One Dataframe column by name.",
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Multiple Dataframe columns by names.",
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "boolean"},
                                        "description": "Boolean mask.",
                                    },
                                    {
                                        "laleType": "callable",
                                        "not": {"type": ["integer", "array", "string"]},
                                        "description": "Callable that is passed the input data X and can return any of the above.",
                                    },
                                ],
                            },
                        ],
                    },
                },
                "remainder": {
                    "description": "Transformation for columns that were not specified in transformers.",
                    "anyOf": [
                        {
                            "description": "Transformer supporting fit and transform.",
                            "laleType": "operator",
                        },
                        {"enum": ["passthrough", "drop"]},
                    ],
                    "default": "drop",
                },
                "sparse_threshold": {
                    "description": """If the output of the different transfromers contains sparse matrices,
these will be stacked as a sparse matrix if the overall density is
lower than this value. Use sparse_threshold=0 to always return dense.""",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                },
                "n_jobs": {
                    "description": "Number of jobs to run in parallel",
                    "anyOf": [
                        {
                            "description": "1 unless in joblib.parallel_backend context.",
                            "enum": [None],
                        },
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of CPU cores.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": None,
                },
                "transformer_weights": {
                    "description": """Multiplicative weights for features per transformer.
The output of the transformer is multiplied by these weights.""",
                    "anyOf": [
                        {
                            "description": "Keys are transformer names, values the weights.",
                            "type": "object",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
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
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {"description": "Target for supervised learning (ignored)."},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """ColumnTransformer_ from scikit-learn applies transformers to columns of an array or pandas DataFrame.

.. _ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.column_transformer.html",
    "import_from": "sklearn.compose",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


ColumnTransformer: lale.operators.PlannedIndividualOp
ColumnTransformer = lale.operators.make_operator(
    sklearn.compose.ColumnTransformer, _combined_schemas
)

if sklearn.__version__ >= "0.21":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.compose.ColumnTransformer.html
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.compose.ColumnTransformer.html
    ColumnTransformer = ColumnTransformer.customize_schema(
        verbose={
            "description": "If True, the time elapsed while fitting each transformer will be printed as it is completed.",
            "type": "boolean",
            "default": False,
        },
    )


lale.docstrings.set_docstrings(ColumnTransformer)
