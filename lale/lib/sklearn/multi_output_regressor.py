# Copyright 2021 IBM Corporation
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

import sklearn.multioutput

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimator",
                "n_jobs",
            ],
            "relevantToOptimizer": [],
            "additionalProperties": False,
            "properties": {
                "estimator": {
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                    "description": "An estimator object implementing `fit` and `predict`.",
                },
                "n_jobs": {
                    "description": "The number of jobs to run in parallel for `fit`, `predict`, and `partial_fit` (if supported by the passed estimator).",
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
            },
        }
    ],
}

_input_fit_schema = {
    "required": ["X", "y"],
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
        "y": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The target values (real numbers).",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Sample weights. If None, then samples are equally weighted. Only supported if the underlying regressor supports sample weights.",
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    },
}

_output_predict_schema = {
    "description": "The predicted regression values.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Multi-output regressor`_ from scikit-learn for multi target regression.

.. _`Multi-output regressor`: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.multi_output_regressor.html",
    "import_from": "sklearn.multioutput",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


MultiOutputRegressor = lale.operators.make_operator(
    sklearn.multioutput.MultiOutputRegressor, _combined_schemas
)

lale.docstrings.set_docstrings(MultiOutputRegressor)
