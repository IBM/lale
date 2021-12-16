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

import pandas as pd

import lale.datasets.data_schemas
import lale.docstrings
import lale.expressions
import lale.operators

try:
    import pyspark.sql
    import pyspark.sql.functions

    spark_installed = True

except ImportError:
    spark_installed = False


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
        if not isinstance(self.columns, dict):
            raise ValueError(
                "Aggregate 'columns' parameter should be of dictionary type."
            )

        agg_info = []
        for new_col_name, expr in self.columns.items():
            if isinstance(expr._expr, ast.Call):
                agg_func_name = expr._expr.func.id  # type: ignore
                old_col_name = lale.expressions._it_column(expr._expr.args[0])
            else:
                agg_func_name = "first"
                old_col_name = lale.expressions._it_column(expr._expr)
            agg_info.append((new_col_name, old_col_name, agg_func_name))
        if isinstance(X, (pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
            aggregated_df = self._transform_pandas(X, agg_info)
        elif isinstance(X, (pyspark.sql.DataFrame, pyspark.sql.GroupedData)):  # type: ignore
            aggregated_df = self._transform_spark(X, agg_info)
        else:
            raise ValueError(f"Unsupported type(X) {type(X)} for Aggregate.")
        named_aggregated_df = lale.datasets.data_schemas.add_table_name(
            aggregated_df, lale.datasets.data_schemas.get_table_name(X)
        )
        return named_aggregated_df

    def _transform_pandas(self, X, agg_info):
        is_grouped = isinstance(X, pd.core.groupby.generic.DataFrameGroupBy)
        if is_grouped:
            _, first_group = next(X.__iter__())  # TODO: what if zero groups?
            value_columns = first_group.columns
        else:
            value_columns = X.columns

        def eval_agg_pandas(old_col_name, agg_func_name):
            if agg_func_name == "collect_set":
                agg_func_name = "unique"
            if is_grouped and old_col_name not in value_columns:
                idx = X.count().index
                if old_col_name not in idx.names:
                    raise KeyError(old_col_name, value_columns, idx.names)
                if agg_func_name != "first":
                    raise ValueError(
                        "Expected plain group-by column access it['{old_col_name}'], found function '{agg_func_name}'"
                    )
                return idx.get_level_values(old_col_name)
            return X[old_col_name].agg(agg_func_name)

        aggregated_columns = {
            new_col_name: eval_agg_pandas(old_col_name, agg_func_name)
            for new_col_name, old_col_name, agg_func_name in agg_info
        }
        if is_grouped:
            aggregated_df = pd.DataFrame(aggregated_columns)
        else:
            aggregated_df = pd.DataFrame.from_records([aggregated_columns])
        return aggregated_df

    def _transform_spark(self, X, agg_info):
        def create_spark_agg_expr(new_col_name, old_col_name, agg_func_name):
            func = getattr(pyspark.sql.functions, agg_func_name)
            result = func(old_col_name).alias(new_col_name)
            return result

        agg_expr = [
            create_spark_agg_expr(new_col_name, old_col_name, agg_func_name)
            for new_col_name, old_col_name, agg_func_name in agg_info
        ]
        aggregated_df = X.agg(*agg_expr)
        keep_columns = [new_col_name for new_col_name, _, _ in agg_info]
        drop_columns = list(set(aggregated_df.columns) - set(keep_columns))
        aggregated_df = aggregated_df.drop(*drop_columns)
        return aggregated_df


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
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.aggregate.html",
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
