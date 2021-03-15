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
import sklearn.preprocessing

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "description": "Scale features using statistics that are robust to outliers.",
    "allOf": [
        {
            "type": "object",
            "required": ["quantile_range", "copy"],
            "relevantToOptimizer": ["with_centering", "with_scaling", "quantile_range"],
            "additionalProperties": False,
            "properties": {
                "with_centering": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, center the data before scaling.",
                },
                "with_scaling": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, scale the data to interquartile range.",
                },
                "quantile_range": {
                    "type": "array",
                    "laleType": "tuple",
                    "minItemsForOptimizer": 2,
                    "maxItemsForOptimizer": 2,
                    "items": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.001,
                            "maximumForOptimizer": 0.3,
                        },
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.7,
                            "maximumForOptimizer": 0.999,
                        },
                    ],
                    "default": [0.25, 0.75],
                    "description": "Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR",
                },
                "copy": {
                    "type": "boolean",
                    "default": True,
                    "description": "If False, try to avoid a copy and do inplace scaling instead.",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Compute the median and quantiles to be used for scaling.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The data used to compute the median and quantiles",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "description": "Center and scale the data.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The data used to scale along the specified axis.",
        },
    },
}
_output_transform_schema = {
    "description": "Center and scale the data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Robust scaler`_ transformer from scikit-learn.

.. _`Robust scaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.robust_scaler.html",
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

RobustScaler: lale.operators.PlannedIndividualOp
RobustScaler = lale.operators.make_operator(
    sklearn.preprocessing.RobustScaler, _combined_schemas
)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.RobustScaler.html
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.preprocessing.RobustScaler.html
    from lale.schemas import Bool

    RobustScaler = RobustScaler.customize_schema(
        unit_variance=Bool(
            desc="If True, scale data so that normally distributed features have a variance of 1. In general, if the difference between the x-values of q_max and q_min for a standard normal distribution is greater than 1, the dataset will be scaled down. If less than 1, the dataset will be scaled up.",
            default=False,
            forOptimizer=True,
        )
    )


lale.docstrings.set_docstrings(RobustScaler)
