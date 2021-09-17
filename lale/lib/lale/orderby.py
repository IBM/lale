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
import ast
from typing import List, Tuple

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
from lale.expressions import Expr
from lale.helpers import (
    _is_ast_attribute,
    _is_ast_subscript,
    _is_pandas_df,
    _is_spark_df,
)


class _OrderByImpl:
    def __init__(self, by=None):
        self.by = by

    def transform_schema(self, s_X):
        return s_X

    def _get_order_key(self, expr_to_parse) -> Tuple[str, bool]:
        order_asc: bool = True
        col: str
        if isinstance(expr_to_parse, Expr):
            expr_to_parse = expr_to_parse._expr
            if isinstance(expr_to_parse, ast.Call):
                op = expr_to_parse.func
                if isinstance(op, ast.Name):
                    name = op.id
                    if name == "asc":
                        order_asc = True
                    elif name == "desc":
                        order_asc = False
                    else:
                        raise ValueError(
                            "OrderBy descriptor expressions must be either asc or desc"
                        )
                else:
                    raise ValueError(
                        "OrderBy expressions must be a string or an order descriptor (asc, desc)"
                    )

                # for now, we only support single argument predicates
                if len(expr_to_parse.args) != 1:
                    raise ValueError(
                        "OrderBy predicates do not support multiple aruguments",
                    )
                arg = expr_to_parse.args[0]
            else:
                arg = expr_to_parse
        else:
            arg = expr_to_parse
        if isinstance(arg, str):
            col = arg
        elif isinstance(arg, ast.Name):
            col = arg.id  # type: ignore
        elif hasattr(ast, "Constant") and isinstance(arg, ast.Constant):
            col = arg.value  # type: ignore
        elif hasattr(ast, "Str") and isinstance(arg, ast.Str):
            col = arg.s
        elif _is_ast_subscript(arg):
            col = arg.slice.value.s  # type: ignore
        elif _is_ast_attribute(arg):
            col = arg.attr  # type: ignore
        else:
            raise ValueError(
                "OrderBy parameters only support string, subscript or dot notation for the column name. For example, it.col_name or it['col_name']."
            )
        return col, order_asc

    def transform(self, X):
        table_name = lale.datasets.data_schemas.get_table_name(X)

        by = self.by
        orders: List[Tuple[str, bool]]
        if isinstance(by, list):
            orders = [self._get_order_key(k) for k in by]
        else:
            orders = [self._get_order_key(by)]

        cols: List[str] = [col for col, _ in orders]
        ascs: List[bool] = [asc for _, asc in orders]
        if _is_pandas_df(X):
            ordered_df = X.sort_values(by=cols, ascending=ascs)
        elif _is_spark_df(X):
            ordered_df = X.orderBy(cols, ascending=ascs)
        else:
            raise ValueError(
                "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
            )

        ordered_df = lale.datasets.data_schemas.add_table_name(ordered_df, table_name)
        return ordered_df


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
                "by": {"description": "OrderBy key(s).", "laleType": "Any"},
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
            "description": "The outer array is over rows.",
            "anyOf": [
                {"laleType": "Any"},
                {
                    "type": "array",
                    "items": {
                        "description": "The inner array is over columns.",
                        "type": "array",
                        "items": {"laleType": "Any"},
                    },
                },
            ],
        }
    },
}

_output_transform_schema = {
    "description": "The outer array is over rows.",
    "anyOf": [
        {
            "type": "array",
            "items": {
                "description": "The inner array is over columns.",
                "type": "array",
                "items": {"laleType": "Any"},
            },
        },
        {"laleType": "Any"},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra OrderBy (sort) operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.orderby.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


OrderBy = lale.operators.make_operator(_OrderByImpl, _combined_schemas)

lale.docstrings.set_docstrings(OrderBy)
