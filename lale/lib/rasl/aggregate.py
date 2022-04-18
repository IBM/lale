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

import numpy as np
import pandas as pd

import lale.docstrings
import lale.expressions
import lale.operators
from lale.datasets.data_schemas import (
    SparkDataFrameWithIndex,
    add_table_name,
    get_table_name,
)

try:
    import pyspark.sql
    import pyspark.sql.functions
    from pyspark.sql.functions import col, isnan, when

    spark_installed = True

except ImportError:
    spark_installed = False


class _AggregateImpl:
    def __init__(self, columns, group_by=[], exclude_value=None):
        self.columns = columns
        self.group_by = group_by
        self.exclude_value = exclude_value

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
        named_aggregated_df = add_table_name(aggregated_df, get_table_name(X))
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
            elif agg_func_name == "mode":
                agg_func_name = (
                    lambda x: x.value_counts()
                    .sort_index(ascending=False)
                    .sort_values(ascending=False)
                    .index[0]
                )  # noqa
            if is_grouped and old_col_name not in value_columns:
                idx = X.count().index
                if old_col_name not in idx.names:
                    raise KeyError(old_col_name, value_columns, idx.names)
                if agg_func_name != "first":
                    raise ValueError(
                        "Expected plain group-by column access it['{old_col_name}'], found function '{agg_func_name}'"
                    )
                return idx.get_level_values(old_col_name)
            X_old_col = X[old_col_name]
            if self.exclude_value is not None:
                return X_old_col[~X_old_col.isin([self.exclude_value])].agg(
                    agg_func_name
                )
            else:
                return X_old_col.agg(agg_func_name)

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
            if agg_func_name == "median":
                agg_func_name = "percentile_approx"
            func = getattr(pyspark.sql.functions, agg_func_name)
            if agg_func_name == "percentile_approx":
                if self.exclude_value is not None:
                    result = func(self._get_exclude_when_expr(old_col_name), 0.5).alias(
                        new_col_name
                    )
                else:
                    result = func(old_col_name, 0.5).alias(new_col_name)
            else:
                if self.exclude_value is not None:
                    result = func(self._get_exclude_when_expr(old_col_name)).alias(
                        new_col_name
                    )
                else:
                    result = func(old_col_name).alias(new_col_name)
            return result

        agg_expr = []
        mode_column_names = []
        for new_col_name, old_col_name, agg_func_name in agg_info:
            if agg_func_name != "mode":
                agg_expr.append(
                    create_spark_agg_expr(new_col_name, old_col_name, agg_func_name)
                )
            else:
                mode_column_names.append((new_col_name, old_col_name))
        if len(agg_expr) == 0:
            # This means that all the aggregate expressions were mode.
            # For that case, compute the mean first, so that the dataframe has the right shape
            # and replace the mean with mode next
            agg_expr = [
                create_spark_agg_expr(new_col_name, old_col_name, "mean")
                for new_col_name, old_col_name, _ in agg_info
            ]

        aggregated_df = X.agg(*agg_expr)
        if len(mode_column_names) > 0:
            if isinstance(X, pyspark.sql.GroupedData):
                raise ValueError(
                    "Mode is not supported as an aggregate immediately after GroupBy for Spark dataframes."
                )
            from pyspark.sql.functions import lit

            for (new_col_name, old_col_name) in mode_column_names:
                if self.exclude_value is not None:
                    if self.exclude_value in [np.nan, "nan"]:
                        filter_expr = ~isnan(old_col_name)
                    else:
                        filter_expr = col(old_col_name) != self.exclude_value
                    aggregated_df = aggregated_df.withColumn(
                        new_col_name,
                        lit(
                            X.filter(filter_expr)
                            .groupby(old_col_name)
                            .count()
                            .orderBy("count", ascending=False)
                            .first()[0]
                        ),
                    )
                else:
                    aggregated_df = aggregated_df.withColumn(
                        new_col_name,
                        lit(
                            X.groupby(old_col_name)
                            .count()
                            .orderBy("count", ascending=False)
                            .first()[0]
                        ),
                    )

        keep_columns = [new_col_name for new_col_name, _, _ in agg_info]
        drop_columns = [col for col in aggregated_df.columns if col not in keep_columns]
        if len(drop_columns) > 0:
            aggregated_df = SparkDataFrameWithIndex(
                aggregated_df, index_names=drop_columns
            )
        return aggregated_df

    def _get_exclude_when_expr(self, col_name):
        if self.exclude_value is not None:
            if self.exclude_value in [np.nan, "nan"]:
                when_expr = when(~isnan(col_name), col(col_name))
            else:
                when_expr = when(
                    col(col_name) != self.exclude_value,
                    col(col_name),
                )
        else:
            when_expr = None
        return when_expr


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
                "exclude_value": {
                    "description": "Exclude this value in computation of aggregates. Useful for missing value imputation.",
                    "laleType": "Any",
                    "default": None,
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
