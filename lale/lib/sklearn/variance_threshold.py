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

import sklearn.feature_selection

import lale.docstrings
import lale.operators

from ._common_schemas import schema_2D_numbers, schema_X_numbers, schema_X_numbers_y_top

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": ["threshold"],
            "relevantToOptimizer": ["threshold"],
            "additionalProperties": False,
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples.",
                    "default": 0,
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                    "distribution": "loguniform",
                },
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`VarianceThreshold`_ transformer from scikit-learn.

.. _`VarianceThreshold`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.normalizer.html",
    "import_from": "sklearn.feature_selection",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": schema_X_numbers_y_top,
        "input_transform": schema_X_numbers,
        "output_transform": schema_2D_numbers,
    },
}

VarianceThreshold = lale.operators.make_operator(
    sklearn.feature_selection.VarianceThreshold, _combined_schemas
)

lale.docstrings.set_docstrings(VarianceThreshold)
