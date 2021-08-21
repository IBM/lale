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

import importlib

import pandas as pd

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators

try:
    import pyspark.sql as pysql
    from pyspark.sql.dataframe import DataFrame as spark_df

    spark_installed = True

except ImportError:
    spark_installed = False

from lale.helpers import _is_ast_attribute, _is_ast_subscript


def _is_pandas_grouped_df(df):
    return isinstance(df, pd.core.groupby.generic.DataFrameGroupBy) or isinstance(
        df, pd.DataFrame
    )


def _is_spark_grouped_df(df):
    if spark_installed:
        return isinstance(df, pysql.GroupedData) or isinstance(df, spark_df)  # type: ignore


class _AggregateImpl:
    def __init__(self, columns, group_by=[]):
        self.columns = columns
        self.group_by = group_by

    # Commented the validation for now to pass the OBM test cases.
    # We can uncomment this when OBM starts supporting the new format of Aggregate operator.
    # @classmethod
    # def validate_hyperparams(cls, group_by=None, **hyperparams):
    #     if group_by is not None:
    #         raise ValueError(
    #             "The use of group_by in Aggregate is deprecated. Please use the GroupBy operator instead."
    #         )

    def transform(self, X):
        agg_info = {}
        agg_expr = {}

        def create_spark_agg_expr(new_col_name, agg_col_func):
            functions_module = importlib.import_module("lale.lib.lale.functions")

            def get_spark_agg_method(agg_method_name):
                return getattr(functions_module, "grouped_" + agg_method_name)

            agg_method = get_spark_agg_method(agg_col_func[1])()  # type: ignore
            return agg_method(agg_col_func[0]).alias(new_col_name)

        if not isinstance(self.columns, dict):
            raise ValueError(
                "Aggregate 'columns' parameter should be of dictionary type."
            )

        for new_col_name, expr in (
            self.columns.items() if self.columns is not None else []
        ):
            agg_func = expr._expr.func.id
            expr_to_parse = expr._expr.args[0]
            if _is_ast_subscript(expr_to_parse):
                agg_col = expr_to_parse.slice.value.s  # type: ignore
            elif _is_ast_attribute(expr_to_parse):
                agg_col = expr_to_parse.attr
            else:
                raise ValueError(
                    "Aggregate 'columns' parameter only supports subscript or dot notation for the key columns. For example, it.col_name or it['col_name']."
                )
            agg_info[new_col_name] = (agg_col, agg_func)
        agg_info_sorted = {
            k: v for k, v in sorted(agg_info.items(), key=lambda item: item[1])
        }

        if _is_pandas_grouped_df(X):
            for agg_col_func in agg_info_sorted.values():
                if agg_col_func[0] in agg_expr:
                    agg_expr[agg_col_func[0]].append(agg_col_func[1])
                else:
                    agg_expr[agg_col_func[0]] = [agg_col_func[1]]
            try:
                aggregated_df = X.agg(agg_expr)
                aggregated_df.columns = agg_info_sorted.keys()
            except KeyError as e:
                raise KeyError(e)
        elif _is_spark_grouped_df(X):
            agg_expr = [
                create_spark_agg_expr(new_col_name, agg_col_func)
                for new_col_name, agg_col_func in agg_info_sorted.items()
            ]
            try:
                aggregated_df = X.agg(*agg_expr)
            except Exception as e:
                raise Exception(e)
        else:
            raise ValueError(
                "Only pandas and spark dataframes are supported by the Aggregate operator."
            )
        named_aggregated_df = lale.datasets.data_schemas.add_table_name(
            aggregated_df, lale.datasets.data_schemas.get_table_name(X)
        )
        return named_aggregated_df


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "columns": {
                    "description": "Aggregations for producing output columns.",
                    "anyOf": [
                        {
                            "description": "Dictionary of output column names and aggregation expressions.",
                            "type": "object",
                            "additionalProperties": {"laleType": "expression"},
                        },
                        {
                            "description": "List of aggregation expressions. The output column name is determined by a heuristic based on the input column name and the transformation function.",
                            "type": "array",
                            "items": {"laleType": "expression"},
                        },
                    ],
                    "default": [],
                },
                "group_by": {
                    "description": "Group by columns for aggregates.",
                    "anyOf": [
                        {
                            "description": "Expressions for columns name if there is a single column.",
                            "laleType": "expression",
                        },
                        {
                            "description": "List of expressions for columns.",
                            "type": "array",
                            "items": {"laleType": "expression"},
                        },
                    ],
                    "default": [],
                },
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
            "description": "Output of the group by operator - Pandas / Pyspark grouped dataframe",
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
            "minItems": 1,
        }
    },
}

_output_transform_schema = {
    "description": "The outer array is over rows.",
    "type": "array",
    "items": {
        "description": "The inner array is over columns.",
        "type": "array",
        "items": {"laleType": "Any"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra aggregate operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.aggregate.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Aggregate = lale.operators.make_operator(_AggregateImpl, _combined_schemas)

lale.docstrings.set_docstrings(Aggregate)
