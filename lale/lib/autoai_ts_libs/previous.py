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

import numpy as np
from autoai_ts_libs.transforms.imputers import (  # type: ignore # noqa
    previous as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "append_tranform",
                "ts_icol_loc",
                "ts_ocol_loc",
                "missing_val_identifier",
                "default_value",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "append_tranform": {
                    "description": """When set to True applies transformation on the last column and appends all
original columns to the right of the transformed column. So output will have columns in order,
                input:
                timestamp, col_a
                output
                timestamp, transformed_col_a, col_a
                input:
                timestamp, col_a, col_b
                output
                timestamp, transformed_col_b, col_a, col_b
If append_tranform is set to False, on transformed column along with timestamp will be returned and original columns
will be dropped.
                input:
                timestamp, col_a
                output
                timestamp, transformed_col_a""",
                    "type": "boolean",
                    "default": False,
                },
                "ts_icol_loc": {
                    "description": """This parameter tells the forecasting modeling the absolute location of the timestamp column.
For specifying time stamp location put value in array e.g., [0] if 0th column is time stamp. The array is to support
multiple timestamps in future. If ts_icol_loc = -1 that means no timestamp is provided and all data is
time series. With ts_icol_loc=-1, the model will assume all the data is ordered and equally sampled.""",
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {"enum": [-1]},
                    ],
                    "default": -1,
                },
                "ts_ocol_loc": {
                    "description": """This parameter tells the interpolator the absolute location of the timestamp column location in the output of the
interpolator, if set to -1 then timestamp will not be includeded otherwise it will be included on specified column
number. if ts_ocol_loc is specified outside range i.e., < 0 or > 'total number of columns' then timestamp is
appended at 0 i.e., the first column in the output of the interpolator.""",
                    "type": "integer",
                    "default": -1,
                },
                "missing_val_identifier": {
                    "description": "Missing value to be imputed.",
                    "laleType": "Any",  # TODO:refine type
                    "default": np.nan,
                },
                "default_value": {
                    "description": """This is the default value that will be used by interpolator in cases where it is needed,
e.g., in case of fill interpolator it is filled with default_value.""",
                    "type": "number",
                    "default": 0,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
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
        },
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
    "description": """Operator from `autoai_ts_libs`_.

.. _`autoai_ts_libs`: https://pypi.org/project/autoai-ts-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.previous.html",
    "import_from": "autoai_ts_libs.transforms.imputers",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "imputer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

previous = lale.operators.make_operator(model_to_be_wrapped, _combined_schemas)

lale.docstrings.set_docstrings(previous)
