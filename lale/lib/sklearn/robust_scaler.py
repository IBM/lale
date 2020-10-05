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

import sklearn.preprocessing

import lale.docstrings
import lale.operators


class RobustScalerImpl:
    def __init__(
        self,
        with_centering=True,
        with_scaling=True,
        quantile_range=(0.25, 0.75),
        copy=None,
    ):
        self._hyperparams = {
            "with_centering": with_centering,
            "with_scaling": with_scaling,
            "quantile_range": quantile_range,
            "copy": copy,
        }
        self._wrapped_model = sklearn.preprocessing.RobustScaler(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


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
            "items": {"type": "array", "items": {"type": "number"},},
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
            "items": {"type": "array", "items": {"type": "number"},},
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

lale.docstrings.set_docstrings(RobustScalerImpl, _combined_schemas)

RobustScaler = lale.operators.make_operator(RobustScalerImpl, _combined_schemas)
