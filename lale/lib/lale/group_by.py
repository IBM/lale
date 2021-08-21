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

import numpy as np

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
from lale.helpers import (
    _is_ast_attribute,
    _is_ast_subscript,
    _is_pandas_df,
    _is_spark_df,
)


class _GroupByImpl:
    def __init__(self, by=None):
        self.by = by

    # Parse the 'by' element passed as input
    def _get_group_key(self, expr_to_parse):
        if _is_ast_subscript(expr_to_parse):
            return expr_to_parse.slice.value.s  # type: ignore
        elif _is_ast_attribute(expr_to_parse):
            return expr_to_parse.attr
        else:
            raise ValueError(
                "GroupBy by parameter only supports subscript or dot notation for the key columns. For example, it.col_name or it['col_name']."
            )

    def transform(self, X):
        group_by_keys = []
        for by_element in self.by if self.by is not None else []:
            expr_to_parse = by_element._expr
            group_by_keys.append(self._get_group_key(expr_to_parse))
        col_not_in_X = np.setdiff1d(group_by_keys, X.columns)
        if col_not_in_X.size > 0:
            raise ValueError(
                "GroupBy key columns {} not present in input dataframe X.".format(
                    col_not_in_X
                )
            )
        if _is_spark_df(X):
            grouped_df = X.groupby(group_by_keys)
        elif _is_pandas_df(X):
            grouped_df = X.groupby(group_by_keys, sort=False)
        else:
            raise ValueError(
                "Only pandas and spark dataframes are supported by the GroupBy operator."
            )
        named_grouped_df = lale.datasets.data_schemas.add_table_name(
            grouped_df, lale.datasets.data_schemas.get_table_name(X)
        )
        return named_grouped_df


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["by"],
            "relevantToOptimizer": [],
            "properties": {
                "by": {"description": "GroupBy key(s).", "laleType": "Any"},
            },
        }
    ]
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "List of tables.",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
            "minItems": 1,
        }
    },
}

_output_transform_schema = {
    "description": "Features; no restrictions on data type.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra group_by operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.group_by.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


GroupBy = lale.operators.make_operator(_GroupByImpl, _combined_schemas)

lale.docstrings.set_docstrings(GroupBy)
