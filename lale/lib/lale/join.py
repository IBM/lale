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

import pandas as pd

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
from lale.helpers import _is_ast_attribute, _is_ast_subscript, _is_df, _is_spark_df

try:
    from pyspark.sql.functions import col

    spark_installed = True

except ImportError:
    spark_installed = False


class _JoinImpl:
    def __init__(
        self,
        *,
        pred=None,
        join_limit=None,
        sliding_window_length=None,
        join_type="inner",
        name=None,
    ):
        self.pred = pred
        self.join_type = join_type
        self.name = name

    # Parse the predicate element passed as input
    @classmethod
    def _get_join_info(cls, expr_to_parse):
        left_key = []
        right_key = []
        if _is_ast_subscript(expr_to_parse.left.value):
            left_name = expr_to_parse.left.value.slice.value.s  # type: ignore
        elif _is_ast_attribute(expr_to_parse.left.value):
            left_name = expr_to_parse.left.value.attr
        else:
            raise ValueError(
                "ERROR: Expression type not supported! Formats supported: it.table_name.column_name or it['table_name'].column_name"
            )
        if _is_ast_subscript(expr_to_parse.left):
            left_key.append(expr_to_parse.left.slice.value.s)  # type: ignore
        elif _is_ast_attribute(expr_to_parse.left):
            left_key.append(expr_to_parse.left.attr)
        else:
            raise ValueError(
                "ERROR: Expression type not supported! Formats supported: it.table_name.column_name or it.table_name['column_name']"
            )
        if _is_ast_subscript(expr_to_parse.comparators[0].value):
            right_name = expr_to_parse.comparators[0].value.slice.value.s  # type: ignore
        elif _is_ast_attribute(expr_to_parse.comparators[0].value):
            right_name = expr_to_parse.comparators[0].value.attr
        else:
            raise ValueError(
                "ERROR: Expression type not supported! Formats supported: it.table_name.column_name or it['table_name'].column_name"
            )
        if _is_ast_subscript(expr_to_parse.comparators[0]):
            right_key.append(expr_to_parse.comparators[0].slice.value.s)  # type: ignore
        elif _is_ast_attribute(expr_to_parse.comparators[0]):
            right_key.append(expr_to_parse.comparators[0].attr)
        else:
            raise ValueError(
                "ERROR: Expression type not supported! Formats supported: it.table_name.column_name or it.table_name['column_name']"
            )
        return left_name, left_key, right_name, right_key

    @classmethod
    def validate_hyperparams(cls, pred=None, **hyperparams):
        tables_encountered = set()

        for key in pred:
            if isinstance(key, list):
                sub_list_tables = (
                    list()
                )  # use an ordered list to improve error messages
                for sub_key in key:
                    (
                        left_table_name,
                        left_key_col,
                        right_table_name,
                        right_key_col,
                    ) = cls._get_join_info(sub_key._expr)
                    if sub_list_tables and not (
                        left_table_name in sub_list_tables
                        and right_table_name in sub_list_tables
                    ):
                        sub_list_tables.append(left_table_name)
                        first_table_names = ", ".join(sub_list_tables)
                        raise ValueError(
                            "ERROR: Composite key involving the {}, and {} tables is problematic, since it references more than two tables.".format(
                                first_table_names, right_table_name
                            )
                        )
                    elif tables_encountered and not (
                        left_table_name in tables_encountered
                        or right_table_name in tables_encountered
                    ):
                        left_expr = f"it.{left_table_name}{left_key_col}"
                        right_expr = f"it.{right_table_name}{right_key_col}"
                        raise ValueError(
                            "ERROR: Composite key involving {} == {} is problematic, since neither the {} nor the {} tables were used in a previous key. Join operations must be chained (they can't have two disconnected join conditions)".format(
                                left_expr, right_expr, left_table_name, right_table_name
                            )
                        )
                    sub_list_tables.append(left_table_name)
                    sub_list_tables.append(right_table_name)
                    tables_encountered.add(left_table_name)
                    tables_encountered.add(right_table_name)
            else:
                (
                    left_table_name,
                    left_key_col,
                    right_table_name,
                    right_key_col,
                ) = cls._get_join_info(key._expr)
                if tables_encountered and not (
                    left_table_name in tables_encountered
                    or right_table_name in tables_encountered
                ):
                    left_expr = f"it.{left_table_name}{left_key_col}"
                    right_expr = f"it.{right_table_name}{right_key_col}"
                    raise ValueError(
                        "ERROR: Single key involving {} == {} is problematic, since neither the {} nor the {} tables were used in a previous key. Join operations must be chained (they can't have two disconnected join conditions)".format(
                            left_expr, right_expr, left_table_name, right_table_name
                        )
                    )
                tables_encountered.add(left_table_name)
                tables_encountered.add(right_table_name)

    def transform(self, X):
        # X is assumed to be a list of datasets with get_table_name(d) != None
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
                        col("{}.{}".format("left_table", key)).eqNullSafe(
                            col("{}.{}".format("right_table", right_key_col[k]))
                        )
                    )
                    if key == right_key_col[k]:
                        drop_col.append(key)
                op_df = left_table.join(right_table, on, self.join_type)
                for key in drop_col:
                    op_df = op_df.drop(getattr(right_table, key))
                return op_df

            # Joining pandas dataframes
            op_df = pd.merge(
                left_df,
                right_df,
                how=self.join_type,
                left_on=left_key_col,
                right_on=right_key_col,
            )
            return op_df

        def fetch_one_df(named_df, table_name):
            if lale.datasets.data_schemas.get_table_name(named_df) == table_name:
                return named_df
            return None

        def fetch_df(left_table_name, right_table_name):
            left_df = []
            right_df = []
            for named_df in X:
                if not tables_encountered:
                    left_df_candidate = fetch_one_df(named_df, left_table_name)
                    if _is_df(left_df_candidate):
                        left_df = left_df_candidate
                    right_df_candidate = fetch_one_df(named_df, right_table_name)
                    if _is_df(right_df_candidate):
                        right_df = right_df_candidate
                else:
                    if left_table_name in tables_encountered:
                        left_df = joined_df
                        right_df_candidate = fetch_one_df(named_df, right_table_name)
                        if _is_df(right_df_candidate):
                            right_df = right_df_candidate
                    elif right_table_name in tables_encountered:
                        right_df = joined_df
                        left_df_candidate = fetch_one_df(named_df, left_table_name)
                        if _is_df(left_df_candidate):
                            left_df = left_df_candidate
            return left_df, right_df

        # Iterate over all the elements of the predicate
        for pred_element in self.pred if self.pred is not None else []:
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
            if not _is_df(left_df) or not _is_df(right_df):
                raise ValueError(
                    "ERROR: Cannot perform join operation, either '{}' or '{}' table not present in input X!".format(
                        left_table_name, right_table_name
                    )
                )
            columns_in_both_tables = set(left_df.columns).intersection(  # type: ignore
                set(right_df.columns)  # type: ignore
            )
            if columns_in_both_tables and not set(
                sorted(columns_in_both_tables)
            ) == set(sorted(left_key_col + right_key_col)):
                raise ValueError(
                    "Cannot perform join operation! Non-key columns cannot be duplicate."
                )
            joined_df = join_df(left_df, right_df)
            tables_encountered.add(left_table_name)
            tables_encountered.add(right_table_name)
        return lale.datasets.data_schemas.add_table_name(joined_df, self.name)

    def viz_label(self) -> str:
        if isinstance(self.name, str):
            return f"Join:\n{self.name}"
        return "Join"


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "pred",
                "join_limit",
                "sliding_window_length",
                "join_type",
                "name",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "pred": {
                    "description": "Join predicate. Given as Python AST expression.",
                    "laleType": "Any",
                },
                "join_limit": {
                    "description": """Not yet implemented!
For join paths that are one-to-many, join_limit is use to sample the joined results.
When the right hand side of the join has a timestamp column, the join_limit is applied to select the most recent rows.
When the right hand side does not have a timestamp, it randomly samples join_limit number of rows.
Sampling is applied after each pair of tables are joined.""",
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                },
                "sliding_window_length": {
                    "description": """Not yet implemented!
sliding_window_length is also used for sampling the joined results,
only rows in a recent window of length sliding_window_length seconds is used in addition to join_limit.""",
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                },
                "join_type": {
                    "description": """There are various types of SQL joins available and join_type gives the user the option
to choose which type of join the user wants to implement.""",
                    "enum": ["inner", "left", "right"],
                    "default": "inner",
                },
                "name": {
                    "description": "The table name to be given to the output dataframe.",
                    "anyOf": [
                        {
                            "type": "string",
                            "pattern": "[^ ]",
                            "description": "String (cannot be all spaces).",
                        },
                        {
                            "enum": [None],
                            "description": "No table name.",
                        },
                    ],
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
