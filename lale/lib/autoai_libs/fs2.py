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

import autoai_libs.cognito.transforms.transform_utils

import lale.docstrings
import lale.operators


class _FS2Impl:
    def __init__(
        self, cols_ids_must_keep, additional_col_count_to_keep, ptype, eval_algo
    ):
        self._hyperparams = {
            "cols_ids_must_keep": cols_ids_must_keep,
            "additional_col_count_to_keep": additional_col_count_to_keep,
            "ptype": ptype,
            "eval_algo": eval_algo,
        }
        self._wrapped_model = autoai_libs.cognito.transforms.transform_utils.FS2(
            **self._hyperparams
        )

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        result = self._wrapped_model.transform(X)
        try:
            if hasattr(self, "column_names"):
                self.column_names = [
                    self.column_names[i]
                    for i in self._wrapped_model.cols_to_keep_final_
                ]
            if hasattr(self, "column_dtypes"):
                self.column_dtypes = [
                    self.column_dtypes[i]
                    for i in self._wrapped_model.cols_to_keep_final_
                ]
        except Exception:
            pass
        return result

    def set_meta_data(self, meta_data_dict):
        if "column_names" in meta_data_dict.keys():
            self.column_names = meta_data_dict["column_names"]
        if "column_dtypes" in meta_data_dict.keys():
            self.column_dtypes = meta_data_dict["column_dtypes"]

    def get_transform_meta_output(self):
        return_dict = {}
        if hasattr(self, "column_names"):
            return_dict["column_names"] = self.column_names
        if hasattr(self, "column_dtypes"):
            return_dict["column_dtypes"] = self.column_dtypes
        return return_dict


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "cols_ids_must_keep",
                "additional_col_count_to_keep",
                "ptype",
                "eval_algo",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "cols_ids_must_keep": {
                    "description": "Serial numbers of the columns that must be kept irrespective of their feature importance.",
                    "laleType": "Any",  # Found a value `range(0, 20) in fs1`
                    "default": [],
                },
                "additional_col_count_to_keep": {
                    "description": "How many columns need to be retained.",
                    "type": "integer",
                    "minimum": 0,
                },
                "ptype": {
                    "description": "Problem type.",
                    "enum": ["classification", "regression"],
                    "default": "classification",
                },
                "eval_algo": {
                    "description": "A supervised model where fit() sets `feature_importances_`.",
                    "laleType": "Any",
                    "default": None,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
        "y": {
            "type": "array",
            "items": {"laleType": "Any"},
            "description": "Target values.",
        },
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}}
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {"type": "array", "items": {"laleType": "Any"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Feature selection, type 2.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.fs2.html",
    "import_from": "autoai_libs.cognito.transforms.transform_utils",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


FS2 = lale.operators.make_operator(_FS2Impl, _combined_schemas)

lale.docstrings.set_docstrings(FS2)
