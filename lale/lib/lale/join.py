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
import lale.docstrings
import lale.operators

try:
    from pyspark.sql.dataframe import DataFrame as spark_df
    spark_installed = True
except ImportError:
    spark_installed = False

def _is_df(d):
    return isinstance(d, pd.DataFrame) or isinstance(d, pd.Series) or isinstance(d, spark_df)

def _is_spark_df(d):
    return isinstance(d, spark_df)

class _JoinImpl:

    def __init__(self, pred, join_type, join_limit, sliding_window_length = None):
        self.pred = pred
        self.join_type = join_type
        self.join_limit = join_limit
        self.sliding_window_length = sliding_window_length

    def transform(self, X):
        # X is assumed to be a list of dictionaries
        joined_df = pd.DataFrame()
        tables_encountered = set()
        counter = 0

        # Incorporate join type handling
        # Valdiation of predicate
        # Composite key joins
        # Spark df joins

        def join_pd(left_df, right_df, pred_start, pred_end):
            if _is_spark_df(left_df) and _is_spark_df(right_df):
                on = []
                #if len(left_key_col) == len(right_key_col):
                    #for i in left_key_col
                op_df = left_df.join(right_df, on, how = self.join_type)
            if _is_df(left_df) and _is_df(right_df):
                op_df = pd.merge(left_df, right_df,  how = self.join_type, left_on = left_key_col, right_on = right_key_col)
            else:
                raise ValueError('One of the tables to be joined not present in input!')
            return op_df


        def get_join_info(expr_to_parse):
            left_key = []; right_key = []
            left_name = expr_to_parse.left.value.attr
            left_key.append(expr_to_parse.left.attr)
            right_name = expr_to_parse.comparators[0].value.attr
            right_key.append(expr_to_parse.comparators[0].attr)
            return left_name, left_key, right_name, right_key


        for i, p in enumerate(self.pred):
            # import pdb;pdb.set_trace()
            # dir(p._expr.left)

            if counter > 0:
                counter -= 1
                continue
            left_table_name, left_key_col, right_table_name, right_key_col = get_join_info(p._expr)

            j = i + 1
            while j < len(self.pred):
                temp_left_name, temp_left_key, temp_right_name, temp_right_key = get_join_info(self.pred[j]._expr)
                if not (left_table_name == temp_left_name and right_table_name == temp_right_name):
                    break
                left_key_col.extend(temp_left_key)
                right_key_col.extend(temp_right_key)
                j += 1
                counter += 1

            for a_dict in X:
                if not tables_encountered:
                    if _is_df(a_dict.get(left_table_name)):
                        left_df = a_dict.get(left_table_name)
                    if _is_df(a_dict.get(right_table_name)):
                        right_df = a_dict.get(right_table_name)
                else:
                    if left_table_name in tables_encountered and right_table_name in tables_encountered:
                        print('Composite Keys Scenario')
                        # raise ValueError('Composite key join conditions in the predicate should be adjacent to each other!')
                    elif left_table_name in tables_encountered:
                        left_df = joined_df
                        if _is_df(a_dict.get(right_table_name)):
                            right_df = a_dict.get(right_table_name)
                    elif right_table_name in tables_encountered:
                        right_df = joined_df
                        if _is_df(a_dict.get(left_table_name)):
                            left_df = a_dict.get(left_table_name)
                    else:
                        raise ValueError('Join conditions in the predicate should be chained!')

            joined_df = join_pd(left_df, right_df, i, counter + 1)
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
