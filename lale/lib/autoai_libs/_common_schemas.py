# Copyright 2023 IBM Corporation
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

"""This file contains common schema fragments used in the autoai_libs schemas
"""

from typing import Any, Dict

JSON_TYPE = Dict[str, Any]

# activate_flag
_hparam_activate_flag_unmodified: JSON_TYPE = {
    "description": "If False, transform(X) outputs the input numpy array X unmodified.",
    "type": "boolean",
    "default": True,
}

_hparam_activate_flag_features: JSON_TYPE = {
    "description": "If False, the features are not generated.",
    "type": "boolean",
    "default": True,
}

_hparam_activate_flag_active: JSON_TYPE = {
    "description": "Determines whether transformer is active or not.",
    "type": "boolean",
    "default": True,
}

_hparam_col_dtypes: JSON_TYPE = {
    "description": "List of the datatypes of the feature columns.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"enum": [None]},
    ],
    "default": None,
}

_hparam_dtypes_list: JSON_TYPE = {
    "anyOf": [
        {
            "description": "Strings that denote the type of each column of the input numpy array X.",
            "type": "array",
            "items": {
                "enum": [
                    "char_str",
                    "int_str",
                    "float_str",
                    "float_num",
                    "float_int_num",
                    "int_num",
                    "boolean",
                    "Unknown",
                    "missing",
                ]
            },
        },
        {
            "description": "If None, the column types are discovered.",
            "enum": [None],
        },
    ],
    "default": None,
}

_hparam_sklearn_version_family: JSON_TYPE = {
    "description": "The sklearn version for backward compatibility with versions 019 and 020dev. Currently unused.",
    "enum": ["20", "21", "22", "23", "24", None, "1"],
    "default": None,
}


def _hparam_column_headers_list(description: str) -> JSON_TYPE:
    return {
        "description": description,
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "array", "items": {"type": "integer"}},
            {"enum": [None]},
        ],
        "default": None,
    }


_hparam_fs_cols_ids_must_keep: JSON_TYPE = {
    "description": "Serial numbers of the columns that must be kept irrespective of their feature importance.",
    "laleType": "Any",  # Found a value `range(0, 20)`
    "default": [],
}

_hparams_fs_additional_col_count_to_keep: JSON_TYPE = {
    "description": "How many columns need to be retained.",
    "type": "integer",
    "minimum": 0,
}

_hparams_fs_ptype: JSON_TYPE = {
    "description": "Problem type.",
    "enum": ["classification", "regression"],
    "default": "classification",
}


def _hparams_column_index_list(description: str) -> JSON_TYPE:
    return {
        "description": description,
        "anyOf": [
            {"type": "array", "items": {"type": "integer", "minimum": 0}},
            {"enum": [None]},
        ],
        "default": None,
    }


_hparams_transformer_name: JSON_TYPE = {
    "description": "A string name that uniquely identifies this transformer from others.",
    "anyOf": [{"type": "string"}, {"enum": [None]}],
    "default": None,
}


def _hparams_fun_pointer(description: str) -> JSON_TYPE:
    return {
        "description": description,
        "laleType": "Any",
        "default": None,
    }


_hparams_datatype_spec: JSON_TYPE = {"type": "array", "items": {"type": "string"}}


def _hparams_datatypes(description: str) -> JSON_TYPE:
    return {
        "description": description,
        "anyOf": [
            _hparams_datatype_spec,
            {"enum": [None]},
        ],
        "default": None,
    }


def _hparams_feat_constraints(description: str) -> JSON_TYPE:
    return {
        "description": description,
        "laleType": "Any",
        "default": None,
    }


_hparams_tgraph: JSON_TYPE = {
    "description": "Should be the invoking TGraph() object.",
    "anyOf": [
        {"laleType": "Any"},
        {
            "enum": [None],
            "description": "Passing None will result in some failure to detect some inefficiencies due to lack of caching.",
        },
    ],
    "default": None,
}

_hparams_apply_all: JSON_TYPE = {
    "description": "Only use applyAll = True. It means that the transformer will enumerate all features (or feature sets) that match the specified criteria and apply the provided function to each.",
    "type": "boolean",
    "default": True,
}

_hparams_col_names: JSON_TYPE = {
    "description": "Names of the feature columns in a list.",
    "anyOf": [
        {"type": "array", "items": {"type": "string"}},
        {"enum": [None]},
    ],
    "default": None,
}

_hparams_col_as_json_objects: JSON_TYPE = {
    "description": "Names of the feature columns in a json dict.",
    "anyOf": [{"type": "object"}, {"enum": [None]}],
    "default": None,
}

_hparams_tans_class: JSON_TYPE = {
    "description": "A class that implements fit() and transform() in accordance with the transformation function definition.",
    "laleType": "Any",
    "default": None,
}
