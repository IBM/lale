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

import lale.docstrings
import lale.operators

try:
    from pyspark.sql.dataframe import DataFrame as spark_df
    from pyspark.sql.functions import col

    spark_installed = True

except ImportError:
    spark_installed = False


def _is_df(d):
    return (
        isinstance(d, pd.DataFrame)
        or isinstance(d, pd.Series)
        or isinstance(d, spark_df)
    )


def _is_spark_df(d):
    return isinstance(d, spark_df)


class _JoinImpl:
    def __init__(self, pred, join_type, join_limit, sliding_window_length=None):
        self.pred = pred
        self.join_type = join_type
        self.join_limit = join_limit
        self.sliding_window_length = sliding_window_length
        self._validate_predicate()

    # Parse the predicate element passed as input
    def _get_join_info(self, expr_to_parse):
        left_key = []
        right_key = []
        left_name = expr_to_parse.left.value.attr
        if isinstance(expr_to_parse.left, ast.Subscript):
            left_key.append(expr_to_parse.left.slice.value.s)  # type: ignore
        elif isinstance(expr_to_parse.left, ast.Attribute):
            left_key.append(expr_to_parse.left.attr)
        else:
            raise ValueError(
                "Expression type not supported! Formats supported: it.tbl_name.col_name or it.tbl_name['col_name']"
            )
        right_name = expr_to_parse.comparators[0].value.attr
        if isinstance(expr_to_parse.comparators[0], ast.Subscript):
            right_key.append(expr_to_parse.comparators[0].slice.value.s)  # type: ignore
        elif isinstance(expr_to_parse.comparators[0], ast.Attribute):
            right_key.append(expr_to_parse.comparators[0].attr)
        else:
            raise ValueError(
                "Expression type not supported! Formats supported: it.tbl_name.col_name or it.tbl_name['col_name']"
            )
        return left_name, left_key, right_name, right_key

    def _validate_predicate(self):
        tables_encountered = set()
        for key in self.pred:
            if isinstance(key, list):
                sub_list_tables = set()
                for sub_key in key:
                    (
                        left_table_name,
                        left_key_col,
                        right_table_name,
                        right_key_col,
                    ) = self._get_join_info(sub_key._expr)
                    if sub_list_tables and not (
                        left_table_name in sub_list_tables
                        and right_table_name in sub_list_tables
                    ):
                        raise ValueError(
                            "One of the composite keys tried joining more than two tables!"
                        )
                    elif (
                        sub_list_tables
                        and tables_encountered
                        and not (
                            left_table_name in tables_encountered
                            or right_table_name in tables_encountered
                        )
                    ):
                        raise ValueError(
                            "One of the composite keys involve an unused table! Join operations should be chained!"
                        )
                    sub_list_tables.add(left_table_name)
                    sub_list_tables.add(right_table_name)
                    tables_encountered.add(left_table_name)
                    tables_encountered.add(right_table_name)
            else:
                (
                    left_table_name,
                    left_key_col,
                    right_table_name,
                    right_key_col,
                ) = self._get_join_info(key._expr)
                if tables_encountered and not (
                    left_table_name in tables_encountered
                    or right_table_name in tables_encountered
                ):
                    raise ValueError(
                        "One of the single keys involve an unused table! Join operations should be chained!"
                    )
                tables_encountered.add(left_table_name)
                tables_encountered.add(right_table_name)

    def transform(self, X):
        # X is assumed to be a list of dictionaries
        joined_df = pd.DataFrame()
        tables_encountered = set()

        # Implementation of join operator
        def join_df(left_df, right_df):

            # Joining spark dataframes
            if _is_spark_df(left_df) and _is_spark_df(right_df):
                on = []
                drop_col = []
                left_table = left_df.alias("left_table")
                right_table = right_df.alias("right_table")
                for k, key in enumerate(left_key_col):
                    on.append(
                        col("{}.{}".format("left_table", key))
                        == col("{}.{}".format("right_table", right_key_col[k]))
                    )
                    if key == right_key_col[k]:
                        drop_col.append(key)
                op_df = left_table.join(right_table, on, self.join_type)
                for key in drop_col:
                    op_df = op_df.drop(getattr(right_table, key))

            # Joining pandas dataframes
            elif _is_df(left_df) and _is_df(right_df):
                op_df = pd.merge(
                    left_df,
                    right_df,
                    how=self.join_type,
                    left_on=left_key_col,
                    right_on=right_key_col,
                )
            else:
                raise ValueError("One of the tables to be joined not present in input!")
            return op_df

        def fetch_df(left_table_name, right_table_name):
            left_df = []
            right_df = []
            for a_dict in X:
                if not tables_encountered:
                    if _is_df(a_dict.get(left_table_name)):
                        left_df = a_dict.get(left_table_name)
                    if _is_df(a_dict.get(right_table_name)):
                        right_df = a_dict.get(right_table_name)
                else:
                    if left_table_name in tables_encountered:
                        left_df = joined_df
                        if _is_df(a_dict.get(right_table_name)):
                            right_df = a_dict.get(right_table_name)
                    elif right_table_name in tables_encountered:
                        right_df = joined_df
                        if _is_df(a_dict.get(left_table_name)):
                            left_df = a_dict.get(left_table_name)
            return left_df, right_df

        # Iterate over all the elements of the predicate
        for pred_element in self.pred:
            left_table_name = ""
            left_key_col = []
            right_table_name = ""
            right_key_col = []
            if isinstance(pred_element, list):
                # Prepare composite key to apply join once for all the participating columns together
                for sub_pred_element in pred_element:
                    (
                        left_table_name,
                        temp_left_key,
                        right_table_name,
                        temp_right_key,
                    ) = self._get_join_info(sub_pred_element._expr)
                    left_key_col.extend(temp_left_key)
                    right_key_col.extend(temp_right_key)
            else:
                (
                    left_table_name,
                    left_key_col,
                    right_table_name,
                    right_key_col,
                ) = self._get_join_info(pred_element._expr)
            left_df, right_df = fetch_df(left_table_name, right_table_name)
            joined_df = join_df(left_df, right_df)
            tables_encountered.add(left_table_name)
            tables_encountered.add(right_table_name)
        return joined_df


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["pred"],
            "relevantToOptimizer": [],
            "properties": {
                "pred": {
                    "description": "Join predicate. Given as Python AST expression.",
                    "laleType": "Any",
                },
                "join_type": {
                    "description": """There are various types of SQL joins available and join_type gives the user the option
to choose which type of join the user wants to implement.""",
                    "anyOf": [{"type": "string"}, {"enum": [None]}],
                    "default": None,
                },
                "join_limit": {
                    "description": """For join paths that are one-to-many, join_limit is use to sample the joined results.
When the right hand side of the join has a timestamp column, the join_limit is applied to select the most recent rows.
When the right hand side does not have a timestamp, it randomly samples join_limit number of rows.
Sampling is applied after each pair of tables are joined.""",
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                },
                "sliding_window_length": {
                    "description": """sliding_window_length is also used for sampling the joined results,
only rows in a recent window of length sliding_window_length seconds is used in addition to join_limit.""",
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
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
    "description": "Relational algebra join operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.join.html",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Join = lale.operators.make_operator(_JoinImpl, _combined_schemas)

lale.docstrings.set_docstrings(Join)
