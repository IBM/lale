# Copyright 2019 IBM Corporation
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

import unittest

import jsonschema
import pandas as pd

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row, SQLContext

    spark_installed = True
except ImportError:
    spark_installed = False

from test import EnableSchemaValidation

from lale import wrap_imported_operators
from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.expressions import (
    count,
    day_of_month,
    day_of_week,
    day_of_year,
    hour,
    it,
    minute,
    month,
    replace,
    string_indexer,
)
from lale.lib.lale import (
    Aggregate,
    ConcatFeatures,
    Filter,
    Hyperopt,
    Join,
    Map,
    Relational,
    Scan,
)
from lale.lib.sklearn import KNeighborsClassifier, LogisticRegression
from lale.operators import make_pipeline_graph


# Testing '==' and '!=' operator with different types of expressions
class TestExpressions(unittest.TestCase):
    def test_expr_1(self):
        with self.assertRaises(TypeError):
            if it.col < 3:
                _ = "If it throws an exception, then the test is successful."

    def test_expr_2(self):
        self.assertFalse(it.col == 5)

    def test_expr_3(self):
        try:
            if it.col == it.col:
                _ = "If it does not throw an exception, then the test is successful."
        except Exception:
            self.fail("Expression 'it.col == it.col' raised an exception unexpectedly!")

    def test_expr_4(self):
        self.assertFalse(it.col == it.col2)

    def test_expr_5(self):
        X = it.col
        self.assertTrue(X == X)

    def test_expr_6(self):
        self.assertFalse(it.col != 5)

    def test_expr_7(self):
        try:
            if it.col != it.col:
                _ = "If it does not throw an exception, then the test is successful."
        except Exception:
            self.fail("Expression 'it.col != it.col' raised an exception unexpectedly!")

    def test_expr_8(self):
        self.assertFalse(it.col != it.col2)

    def test_expr_9(self):
        X = it.col
        self.assertTrue(X != X)


# Testing filter operator for pandas dataframes
class TestFilter(unittest.TestCase):
    # Define pandas dataframes with different structures
    def setUp(self):
        table1 = {
            "train_id": [1, 2, 3, 4, 5],
            "col1": ["NY", "TX", "CA", "NY", "CA"],
            "col2": [0, 1, 1, 0, 1],
        }
        self.df1 = pd.DataFrame(data=table1)

        table3 = {
            "tid": [2, 3, 4, 5],
            "col5": ["Warm", "Cold", "Warm", "Cold"],
        }
        self.df3 = pd.DataFrame(data=table3)

        table5 = {
            "TrainId": [1, 2, 3, 4, 5],
            "col3": ["NY", "NY", "CA", "TX", "TX"],
            "col4": [1, 1, 4, 8, 0],
        }
        self.df5 = pd.DataFrame(data=table5)

        trainable = Join(
            pred=[
                it.main.train_id == it.info.TrainId,
                it.info.TrainId == it.t1.tid,
            ],
            join_type="left",
        )
        self.transformed_df = trainable.transform(
            [{"main": self.df1}, {"info": self.df5}, {"t1": self.df3}]
        )
        self.assertEqual(self.transformed_df.shape, (5, 8))
        self.assertEqual(self.transformed_df["col3"][2], "CA")

    def test_filter_pandas_eq(self):
        trainable = Filter(pred=[it["col3"] == "TX"])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 8))

    def test_filter_pandas_neq(self):
        trainable = Filter(pred=[it["col1"] != it["col3"]])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 8))

    def test_filter_pandas_ge(self):
        trainable = Filter(pred=[it["col4"] >= 5])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (1, 8))

    def test_filter_pandas_gt(self):
        trainable = Filter(pred=[it["train_id"] > it["col4"]])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 8))

    def test_filter_pandas_le(self):
        trainable = Filter(pred=[it["col3"] <= "NY"])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 8))

    def test_filter_pandas_lt(self):
        trainable = Filter(pred=[it["col4"] < it["TrainId"]])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 8))

    def test_filter_pandas_multiple1(self):
        trainable = Filter(pred=[it["col3"] == "TX", it["col4"] > 4])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (1, 8))

    def test_filter_pandas_multiple2(self):
        trainable = Filter(pred=[it["col5"] != "Cold", it["train_id"] <= 4])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 8))

    def test_filter_pandas_multiple3(self):
        trainable = Filter(
            pred=[
                it["train_id"] == it["TrainId"],
                it["col2"] != it["col4"],
                it["col5"] == "Cold",
            ]
        )
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 8))

    def test_filter_pandas_no_col_error(self):
        with self.assertRaises(ValueError):
            trainable = Filter(pred=[it["col_na"] < it["TrainId"]])
            _ = trainable.transform(self.transformed_df)


# Testing filter operator for spark dataframes
class TestFilterSpark(unittest.TestCase):
    # Define spark dataframes with different structures
    def setUp(self):
        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            sqlContext = SQLContext(sc)

            l2 = [
                (1, "NY", 100),
                (2, "NY", 150),
                (3, "TX", 200),
                (4, "TX", 100),
                (5, "CA", 200),
            ]
            rdd = sc.parallelize(l2)
            table2 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col3=x[1], col4=int(x[2]))
            )
            self.spark_df2 = sqlContext.createDataFrame(table2)

            l3 = [(2, "Warm"), (3, "Cold"), (4, "Warm"), (5, "Cold")]
            rdd = sc.parallelize(l3)
            table3 = rdd.map(lambda x: Row(tid=int(x[0]), col5=x[1]))
            self.spark_df3 = sqlContext.createDataFrame(table3)

            l4 = [(1, "NY", 1), (2, "TX", 6), (3, "CA", 2), (4, "NY", 5), (5, "CA", 0)]
            rdd = sc.parallelize(l4)
            table4 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col1=x[1], col2=int(x[2]))
            )
            self.spark_df4 = sqlContext.createDataFrame(table4)

            trainable = Join(
                pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
                join_type="left",
            )
            self.transformed_df = trainable.transform(
                [
                    {"main": self.spark_df4},
                    {"info": self.spark_df2},
                    {"t1": self.spark_df3},
                ]
            )
            self.assertEqual(self.transformed_df.count(), 5)
            self.assertEqual(len(self.transformed_df.columns), 7)
            self.assertEqual(self.transformed_df.collect()[2]["col1"], "CA")

    def test_filter_spark_eq(self):
        if spark_installed:
            trainable = Filter(pred=[it["col3"] == "NY"])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_neq(self):
        if spark_installed:
            trainable = Filter(pred=[it["col1"] != it["col3"]])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 3)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_ge(self):
        if spark_installed:
            trainable = Filter(pred=[it["col4"] >= 150])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 3)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_gt(self):
        if spark_installed:
            trainable = Filter(pred=[it["col2"] > it["tid"]])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_le(self):
        if spark_installed:
            trainable = Filter(pred=[it["col3"] <= "NY"])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 3)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_lt(self):
        if spark_installed:
            trainable = Filter(pred=[it["col2"] < it["TrainId"]])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_multiple1(self):
        if spark_installed:
            trainable = Filter(pred=[it["col3"] == "TX", it["col4"] >= 150])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 1)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_multiple2(self):
        if spark_installed:
            trainable = Filter(pred=[it["col5"] != "Cold", it["TrainId"] <= 4])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_multiple3(self):
        if spark_installed:
            trainable = Filter(
                pred=[
                    it["tid"] == it["TrainId"],
                    it["col2"] > it["tid"],
                    it["col5"] == "Warm",
                ]
            )
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 7)

    def test_filter_spark_no_col_error(self):
        with self.assertRaises(ValueError):
            trainable = Filter(pred=[it["col_na"] < it["TrainId"]])
            _ = trainable.transform(self.transformed_df)


# Testing join operator for pandas dataframes
class TestJoin(unittest.TestCase):
    def test_init(self):
        _ = Join(pred=[it.main.train_id == it.info.TrainId], join_type="inner")

    # Define pandas dataframes with different structures
    def setUp(self):
        table1 = {
            "train_id": [1, 2, 3, 4, 5],
            "col1": ["NY", "TX", "CA", "NY", "CA"],
            "col2": [0, 1, 1, 0, 1],
        }
        self.df1 = pd.DataFrame(data=table1)

        table2 = {
            "TrainId": [1, 2, 3],
            "col3": ["USA", "USA", "UK"],
            "col4": [100, 100, 200],
        }
        self.df2 = pd.DataFrame(data=table2)

        table3 = {
            "tid": [1, 2, 3],
            "col5": ["Warm", "Cold", "Warm"],
        }
        self.df3 = pd.DataFrame(data=table3)

        table4 = {
            "TrainId": [1, 2, 3, 4, 5],
            "col1": ["NY", "TX", "CA", "NY", "CA"],
            "col2": [0, 1, 1, 0, 1],
        }
        self.df4 = pd.DataFrame(data=table4)

        table5 = {
            "TrainId": [1, 2, 3],
            "col3": ["NY", "NY", "CA"],
            "col4": [100, 100, 200],
        }
        self.df5 = pd.DataFrame(data=table5)

        table6 = {
            "t_id": [2, 3],
            "col6": ["USA", "UK"],
        }
        self.df6 = pd.DataFrame(data=table6)

    # Multiple elements in predicate with different key column names
    def test_join_pandas_multiple_left(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="inner",
        )
        transformed_df = trainable.transform(
            [{"main": self.df1}, {"info": self.df2}, {"t1": self.df3}]
        )
        self.assertEqual(transformed_df.shape, (3, 8))
        self.assertEqual(transformed_df["col5"][1], "Cold")

    # Multiple elements in predicate with identical key columns names
    def test_join_pandas_multiple_left1(self):
        trainable = Join(
            pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="left",
        )
        transformed_df = trainable.transform(
            [{"main": self.df4}, {"info": self.df2}, {"t1": self.df3}]
        )
        self.assertEqual(transformed_df.shape, (5, 7))
        self.assertEqual(transformed_df["col3"][2], "UK")

    # Invert one of the join conditions as compared to the test case: test_join_pandas_multiple_left
    def test_join_pandas_multiple_right(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.t1.tid == it.info.TrainId],
            join_type="right",
        )
        transformed_df = trainable.transform(
            [{"main": self.df1}, {"info": self.df2}, {"t1": self.df3}]
        )
        self.assertEqual(transformed_df.shape, (3, 8))
        self.assertEqual(transformed_df["col3"][2], "UK")

    # Composite key join
    def test_join_pandas_composite(self):
        trainable = Join(
            pred=[
                it.t1.tid == it.info.TrainId,
                [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
            ],
            join_type="left",
        )
        transformed_df = trainable.transform(
            [{"main": self.df1}, {"info": self.df5}, {"t1": self.df3}, {"t2": self.df6}]
        )
        self.assertEqual(transformed_df.shape, (5, 8))
        self.assertEqual(transformed_df["col3"][2], "CA")

    # Invert one of the join conditions as compared to the test case: test_join_pandas_composite
    def test_join_pandas_composite1(self):
        trainable = Join(
            pred=[
                [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                it.t1.tid == it.info.TrainId,
                it.t1.tid == it.t2.t_id,
            ],
            join_type="inner",
        )
        transformed_df = trainable.transform(
            [{"main": self.df1}, {"info": self.df5}, {"t1": self.df3}, {"t2": self.df6}]
        )
        self.assertEqual(transformed_df.shape, (1, 10))
        self.assertEqual(transformed_df["col4"][0], 200)

    # Composite key join having conditions involving more than 2 tables
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_pandas_composite_error(self):
        with self.assertRaisesRegex(
            ValueError, "info.*main.*inFo.* more than two tables"
        ):
            trainable = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.inFo.col3],
                    it.t1.tid == it.t2.t_id,
                ],
                join_type="inner",
            )
            _ = trainable.transform(
                [
                    {"main": self.df1},
                    {"info": self.df5},
                    {"t1": self.df3},
                    {"t2": self.df6},
                ]
            )

    # Single joining conditions are not chained
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_pandas_single_error1(self):
        with self.assertRaisesRegex(ValueError, "t3.*t2.* were used"):
            trainable = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                    it.t3.tid == it.t2.t_id,
                ],
                join_type="inner",
            )
            _ = trainable.transform(
                [
                    {"main": self.df1},
                    {"info": self.df5},
                    {"t1": self.df3},
                    {"t2": self.df6},
                ]
            )

    def test_join_pandas_composite_nochain_error(self):
        with self.assertRaisesRegex(ValueError, "t3.*t2.* were used"):
            trainable = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                    [it.t3.tid == it.t2.t_id, it.t3.TrainId == it.t2.TrainId],
                ],
                join_type="inner",
            )
            _ = trainable.transform(
                [
                    {"main": self.df1},
                    {"info": self.df5},
                    {"t1": self.df3},
                    {"t2": self.df6},
                ]
            )

    # Composite key join having conditions involving more than 2 tables
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_pandas_composite_error2(self):
        with self.assertRaisesRegex(
            ValueError, "main.*info.*Main.*inFo.*more than two"
        ):
            trainable = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.Main.col1 == it.inFo.col3],
                    it.t1.tid == it.t2.t_id,
                ],
                join_type="inner",
            )
            _ = trainable.transform(
                [
                    {"main": self.df1},
                    {"info": self.df5},
                    {"t1": self.df3},
                    {"t2": self.df6},
                ]
            )

    # A table to be joinen not present in input X
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_pandas_composite_error3(self):
        with self.assertRaises(ValueError):
            trainable = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                ],
                join_type="inner",
            )
            _ = trainable.transform(
                [
                    {"sample": self.df1},
                    {"info": self.df5},
                    {"t1": self.df3},
                ]
            )

    # TestCase 1: Go_Sales dataset with different forms of predicate (join conditions)
    def test_join_pandas_go_sales1(self):
        go_sales = fetch_go_sales_dataset()
        trainable = Join(
            pred=[
                it.go_daily_sales["Retailer code"]
                == it["go_retailers"]["Retailer code"]
            ],
            join_type="inner",
        )
        transformed_df = trainable.transform(go_sales)
        self.assertEqual(transformed_df.shape, (149257, 10))
        self.assertEqual(transformed_df["Country"][4], "United States")

    # TestCase 2: Go_Sales dataset throws error because of duplicate non-key columns
    def test_join_pandas_go_sales2(self):
        with self.assertRaises(ValueError):
            go_sales = fetch_go_sales_dataset()
            trainable = Join(
                pred=[
                    [
                        it["go_1k"]["Retailer code"]
                        == it.go_daily_sales["Retailer code"],
                        it.go_1k["Product number"]
                        == it["go_daily_sales"]["Product number"],
                    ]
                ],
                join_type="left",
            )
            _ = trainable.transform(go_sales)


# Testing join operator for spark dataframes
class TestJoinSpark(unittest.TestCase):
    # Define spark dataframes with different structures
    def setUp(self):
        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            sqlContext = SQLContext(sc)

            l1 = [(1, "NY", 0), (2, "TX", 1), (3, "CA", 1), (4, "NY", 0), (5, "CA", 1)]
            rdd = sc.parallelize(l1)
            table1 = rdd.map(
                lambda x: Row(train_id=int(x[0]), col1=x[1], col2=int(x[2]))
            )
            self.spark_df1 = sqlContext.createDataFrame(table1)

            l2 = [(1, "USA", 100), (2, "USA", 100), (3, "UK", 200)]
            rdd = sc.parallelize(l2)
            table2 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col3=x[1], col4=int(x[2]))
            )
            self.spark_df2 = sqlContext.createDataFrame(table2)

            l3 = [(1, "Warm"), (2, "Cold"), (3, "Warm")]
            rdd = sc.parallelize(l3)
            table3 = rdd.map(lambda x: Row(tid=int(x[0]), col5=x[1]))
            self.spark_df3 = sqlContext.createDataFrame(table3)

            l4 = [(1, "NY", 0), (2, "TX", 1), (3, "CA", 1), (4, "NY", 0), (5, "CA", 1)]
            rdd = sc.parallelize(l4)
            table4 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col1=x[1], col2=int(x[2]))
            )
            self.spark_df4 = sqlContext.createDataFrame(table4)

            l5 = [(1, "NY", 100), (2, "NY", 100), (3, "CA", 200)]
            rdd = sc.parallelize(l5)
            table5 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col3=x[1], col4=int(x[2]))
            )
            self.spark_df5 = sqlContext.createDataFrame(table5)

            l6 = [(2, "USA"), (3, "UK")]
            rdd = sc.parallelize(l6)
            table6 = rdd.map(lambda x: Row(t_id=int(x[0]), col3=x[1]))
            self.spark_df6 = sqlContext.createDataFrame(table6)

    # Multiple elements in predicate with different key column names
    def test_join_spark_multiple_left(self):
        if spark_installed:
            trainable = Join(
                pred=[
                    it.main.train_id == it.info.TrainId,
                    it.info.TrainId == it.t1.tid,
                ],
                join_type="inner",
            )
            transformed_df = trainable.transform(
                [
                    {"main": self.spark_df1},
                    {"info": self.spark_df2},
                    {"t1": self.spark_df3},
                ]
            )
            self.assertEqual(transformed_df.count(), 3)
            self.assertEqual(len(transformed_df.columns), 8)
            self.assertEqual(transformed_df.collect()[0]["col5"], "Warm")

    # Multiple elements in predicate with identical key columns names
    def test_join_spark_multiple_left1(self):
        if spark_installed:
            trainable = Join(
                pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
                join_type="left",
            )
            transformed_df = trainable.transform(
                [
                    {"main": self.spark_df4},
                    {"info": self.spark_df2},
                    {"t1": self.spark_df3},
                ]
            )
            self.assertEqual(transformed_df.count(), 5)
            self.assertEqual(len(transformed_df.columns), 7)
            self.assertEqual(transformed_df.collect()[2]["col1"], "CA")

    # Invert one of the join conditions as compared to the test case: test_join_spark_multiple_left
    def test_join_spark_multiple_right(self):
        if spark_installed:
            trainable = Join(
                pred=[
                    it.main.train_id == it.info.TrainId,
                    it.t1.tid == it.info.TrainId,
                ],
                join_type="right",
            )
            transformed_df = trainable.transform(
                [
                    {"main": self.spark_df1},
                    {"info": self.spark_df2},
                    {"t1": self.spark_df3},
                ]
            )
            self.assertEqual(transformed_df.count(), 3)
            self.assertEqual(len(transformed_df.columns), 8)
            self.assertEqual(transformed_df.collect()[0]["col2"], 0)

    # Composite key join
    def test_join_spark_composite(self):
        if spark_installed:
            trainable = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                ],
                join_type="left",
            )
            transformed_df = trainable.transform(
                [
                    {"main": self.spark_df1},
                    {"info": self.spark_df5},
                    {"t1": self.spark_df3},
                    {"t2": self.spark_df6},
                ]
            )
            self.assertEqual(transformed_df.count(), 5)
            self.assertEqual(len(transformed_df.columns), 8)
            self.assertEqual(transformed_df.collect()[0]["col2"], 1)

    # Invert one of the join conditions as compared to the test case: test_join_pandas_composite
    def test_join_spark_composite_error_dup_col(self):
        if spark_installed:
            with self.assertRaises(ValueError):
                trainable = Join(
                    pred=[
                        [
                            it.main.train_id == it.info.TrainId,
                            it.main.col1 == it.info.col3,
                        ],
                        it.t1.tid == it.info.TrainId,
                        it.t1.tid == it.t2.t_id,
                    ],
                    join_type="left",
                )
                _ = trainable.transform(
                    [
                        {"main": self.spark_df1},
                        {"info": self.spark_df5},
                        {"t1": self.spark_df3},
                        {"t2": self.spark_df6},
                    ]
                )

    # Composite key join having conditions involving more than 2 tables
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_spark_composite_error(self):
        if spark_installed:
            with self.assertRaises(ValueError):
                trainable = Join(
                    pred=[
                        it.t1.tid == it.info.TrainId,
                        [
                            it.main.train_id == it.info.TrainId,
                            it.main.col1 == it.inFo.col3,
                        ],
                        it.t1.tid == it.t2.t_id,
                    ],
                    join_type="inner",
                )
                _ = trainable.transform(
                    [
                        {"main": self.spark_df1},
                        {"info": self.spark_df5},
                        {"t1": self.spark_df3},
                        {"t2": self.spark_df6},
                    ]
                )

    # Joining conditions are not chained
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_spark_composite_error1(self):
        if spark_installed:
            with self.assertRaises(ValueError):
                trainable = Join(
                    pred=[
                        it.t1.tid == it.info.TrainId,
                        [
                            it.main.train_id == it.info.TrainId,
                            it.main.col1 == it.info.col3,
                        ],
                        it.t3.tid == it.t2.t_id,
                    ],
                    join_type="inner",
                )
                _ = trainable.transform(
                    [
                        {"main": self.spark_df1},
                        {"info": self.spark_df5},
                        {"t1": self.spark_df3},
                        {"t2": self.spark_df6},
                    ]
                )

    # A table to be joined not present in input X
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_spark_composite_error2(self):
        if spark_installed:
            with self.assertRaises(ValueError):
                trainable = Join(
                    pred=[
                        it.t1.tid == it.info.TrainId,
                        [
                            it.main.train_id == it.info.TrainId,
                            it.main.col1 == it.info.col3,
                        ],
                    ],
                    join_type="inner",
                )
                _ = trainable.transform(
                    [
                        {"sample": self.spark_df1},
                        {"info": self.spark_df5},
                        {"t1": self.spark_df3},
                    ]
                )

    # TestCase 1: Go_Sales dataset with different forms of predicate (join conditions)
    def test_join_spark_go_sales1(self):
        if spark_installed:
            go_sales = fetch_go_sales_dataset("spark")
            trainable = Join(
                pred=[
                    it.go_daily_sales["Retailer code"]
                    == it["go_retailers"]["Retailer code"]
                ],
                join_type="inner",
            )
            transformed_df = trainable.transform(go_sales)
            self.assertEqual(transformed_df.count(), 149257)
            self.assertEqual(len(transformed_df.columns), 10)

    # TestCase 2: Go_Sales dataset throws error because of duplicate non-key columns
    def test_join_spark_go_sales2(self):
        if spark_installed:
            with self.assertRaises(ValueError):
                go_sales = fetch_go_sales_dataset("spark")
                trainable = Join(
                    pred=[
                        [
                            it["go_1k"]["Retailer code"]
                            == it.go_daily_sales["Retailer code"],
                            it.go_1k["Product number"]
                            == it["go_daily_sales"]["Product number"],
                        ]
                    ],
                    join_type="left",
                )
                _ = trainable.transform(go_sales)


class TestMap(unittest.TestCase):
    def test_init(self):
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        _ = Map(columns=[replace(it.gender, gender_map), replace(it.state, state_map)])

    def test_transform_replace_list_and_remainder(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        trainable = Map(
            columns=[replace(it.gender, gender_map), replace(it.state, state_map)],
            remainder="drop",
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 2))
        self.assertEqual(transformed_df["gender"][0], "Male")
        self.assertEqual(transformed_df["state"][0], "New York")

    def test_transform_replace_list(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        trainable = Map(
            columns=[replace(it.gender, gender_map), replace(it.state, state_map)]
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 3))
        self.assertEqual(transformed_df["gender"][0], "Male")
        self.assertEqual(transformed_df["state"][0], "New York")

    def test_transform_replace_map(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        trainable = Map(
            columns={
                "new_gender": replace(it.gender, gender_map),
                "new_state": replace(it.state, state_map),
            }
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 3))
        self.assertEqual(transformed_df["new_gender"][0], "Male")
        self.assertEqual(transformed_df["new_state"][0], "New York")

    def test_transform_dom_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-27", "2016-07-26"]})
        trainable = Map(columns=[day_of_month(it.date_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 28)
        self.assertEqual(transformed_df["date_column"][1], 27)
        self.assertEqual(transformed_df["date_column"][2], 26)

    def test_transform_dom_fmt_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-27", "2016-07-26"]})
        trainable = Map(columns=[day_of_month(it.date_column, "%Y-%m-%d")])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 28)
        self.assertEqual(transformed_df["date_column"][1], 27)
        self.assertEqual(transformed_df["date_column"][2], 26)

    def test_transform_dom_fmt_map(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-27", "2016-07-26"]})
        trainable = Map(columns={"dom": day_of_month(it.date_column, "%Y-%m-%d")})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["dom"][0], 28)
        self.assertEqual(transformed_df["dom"][1], 27)
        self.assertEqual(transformed_df["dom"][2], 26)

    def test_transform_dow_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-28", "2016-07-28"]})
        trainable = Map(columns=[day_of_week(it.date_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 5)
        self.assertEqual(transformed_df["date_column"][1], 1)
        self.assertEqual(transformed_df["date_column"][2], 3)

    def test_transform_dow_fmt_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-28", "2016-07-28"]})
        trainable = Map(columns=[day_of_week(it.date_column, "%Y-%m-%d")])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 5)
        self.assertEqual(transformed_df["date_column"][1], 1)
        self.assertEqual(transformed_df["date_column"][2], 3)

    def test_transform_dow_fmt_map(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-28", "2016-07-28"]})
        trainable = Map(columns={"dow": day_of_week(it.date_column, "%Y-%m-%d")})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["dow"][0], 5)
        self.assertEqual(transformed_df["dow"][1], 1)
        self.assertEqual(transformed_df["dow"][2], 3)

    def test_transform_doy_list(self):
        df = pd.DataFrame({"date_column": ["2016-01-01", "2016-06-28", "2016-07-28"]})
        trainable = Map(columns=[day_of_year(it.date_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 1)
        self.assertEqual(transformed_df["date_column"][1], 180)
        self.assertEqual(transformed_df["date_column"][2], 210)

    def test_transform_doy_fmt_list(self):
        df = pd.DataFrame({"date_column": ["2016-01-01", "2016-06-28", "2016-07-28"]})
        trainable = Map(columns=[day_of_year(it.date_column, "%Y-%m-%d")])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 1)
        self.assertEqual(transformed_df["date_column"][1], 180)
        self.assertEqual(transformed_df["date_column"][2], 210)

    def test_transform_doy_fmt_map(self):
        df = pd.DataFrame({"date_column": ["2016-01-01", "2016-06-28", "2016-07-28"]})
        trainable = Map(columns={"doy": day_of_year(it.date_column, "%Y-%m-%d")})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["doy"][0], 1)
        self.assertEqual(transformed_df["doy"][1], 180)
        self.assertEqual(transformed_df["doy"][2], 210)

    def test_transform_hour_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns=[hour(it.date_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 15)
        self.assertEqual(transformed_df["date_column"][1], 12)
        self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_hour_fmt_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns=[hour(it.date_column, "%Y-%m-%d %H:%M:%S")])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 15)
        self.assertEqual(transformed_df["date_column"][1], 12)
        self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_hour_fmt_map(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns={"hour": hour(it.date_column, "%Y-%m-%d %H:%M:%S")})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["hour"][0], 15)
        self.assertEqual(transformed_df["hour"][1], 12)
        self.assertEqual(transformed_df["hour"][2], 1)

    def test_transform_minute_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns=[minute(it.date_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 16)
        self.assertEqual(transformed_df["date_column"][1], 18)
        self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_minute_fmt_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns=[minute(it.date_column, "%Y-%m-%d %H:%M:%S")])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 16)
        self.assertEqual(transformed_df["date_column"][1], 18)
        self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_minute_fmt_map(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns={"minute": minute(it.date_column, "%Y-%m-%d %H:%M:%S")})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["minute"][0], 16)
        self.assertEqual(transformed_df["minute"][1], 18)
        self.assertEqual(transformed_df["minute"][2], 1)

    def test_transform_month_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns=[month(it.date_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 1)
        self.assertEqual(transformed_df["date_column"][1], 6)
        self.assertEqual(transformed_df["date_column"][2], 7)

    def test_transform_month_fmt_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns=[month(it.date_column, "%Y-%m-%d %H:%M:%S")])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date_column"][0], 1)
        self.assertEqual(transformed_df["date_column"][1], 6)
        self.assertEqual(transformed_df["date_column"][2], 7)

    def test_transform_month_fmt_map(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        trainable = Map(columns={"month": month(it.date_column, "%Y-%m-%d %H:%M:%S")})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["month"][0], 1)
        self.assertEqual(transformed_df["month"][1], 6)
        self.assertEqual(transformed_df["month"][2], 7)

    def test_transform_string_indexer_list(self):
        df = pd.DataFrame({"cat_column": ["a", "b", "b"]})
        trainable = Map(columns=[string_indexer(it.cat_column)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["cat_column"][0], 1)
        self.assertEqual(transformed_df["cat_column"][1], 0)
        self.assertEqual(transformed_df["cat_column"][2], 0)

    def test_transform_string_indexer_map(self):
        df = pd.DataFrame({"cat_column": ["a", "b", "b"]})
        trainable = Map(columns={"string_indexed": string_indexer(it.cat_column)})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (3, 1))
        self.assertEqual(transformed_df["string_indexed"][0], 1)
        self.assertEqual(transformed_df["string_indexed"][1], 0)
        self.assertEqual(transformed_df["string_indexed"][2], 0)

    def test_not_expression(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = Map(columns=[123, "hello"])

    def test_with_hyperopt(self):
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        map_replace = Map(
            columns=[replace(it.gender, gender_map), replace(it.state, state_map)],
            remainder="drop",
        )
        pipeline = (
            Relational(
                operator=(Scan(table=it.main) & Scan(table=it.delay)) >> map_replace
            )
            >> LogisticRegression()
        )
        opt = Hyperopt(estimator=pipeline, cv=3, max_evals=5)
        trained = opt.fit(X, y)
        _ = trained

    def test_with_hyperopt2(self):
        from lale.expressions import (
            count,
            it,
            max,
            mean,
            min,
            string_indexer,
            sum,
            variance,
        )

        wrap_imported_operators()
        scan = Scan(table=it["main"])
        scan_0 = Scan(table=it["customers"])
        join = Join(
            pred=[
                (
                    it["main"]["group_customer_id"]
                    == it["customers"]["group_customer_id"]
                )
            ]
        )
        map = Map(
            columns={
                "[main](group_customer_id)[customers]|number_children|identity": it[
                    "number_children"
                ],
                "[main](group_customer_id)[customers]|name|identity": it["name"],
                "[main](group_customer_id)[customers]|income|identity": it["income"],
                "[main](group_customer_id)[customers]|address|identity": it["address"],
                "[main](group_customer_id)[customers]|age|identity": it["age"],
            },
            remainder="drop",
        )
        pipeline_4 = join >> map
        scan_1 = Scan(table=it["purchase"])
        join_0 = Join(
            pred=[(it["main"]["group_id"] == it["purchase"]["group_id"])],
            join_limit=50.0,
        )
        aggregate = Aggregate(
            columns={
                "[main](group_id)[purchase]|price|variance": variance(it["price"]),
                "[main](group_id)[purchase]|time|sum": sum(it["time"]),
                "[main](group_id)[purchase]|time|mean": mean(it["time"]),
                "[main](group_id)[purchase]|time|min": min(it["time"]),
                "[main](group_id)[purchase]|price|sum": sum(it["price"]),
                "[main](group_id)[purchase]|price|count": count(it["price"]),
                "[main](group_id)[purchase]|price|mean": mean(it["price"]),
                "[main](group_id)[purchase]|price|min": min(it["price"]),
                "[main](group_id)[purchase]|price|max": max(it["price"]),
                "[main](group_id)[purchase]|time|max": max(it["time"]),
                "[main](group_id)[purchase]|time|variance": variance(it["time"]),
            },
            group_by=it["row_id"],
        )
        pipeline_5 = join_0 >> aggregate
        map_0 = Map(
            columns={
                "[main]|group_customer_id|identity": it["group_customer_id"],
                "[main]|transaction_id|identity": it["transaction_id"],
                "[main]|group_id|identity": it["group_id"],
                "[main]|comments|identity": it["comments"],
                "[main]|id|identity": it["id"],
                "prefix_0_id": it["prefix_0_id"],
                "next_purchase": it["next_purchase"],
                "[main]|time|identity": it["time"],
            },
            remainder="drop",
        )
        scan_2 = Scan(table=it["transactions"])
        scan_3 = Scan(table=it["products"])
        join_1 = Join(
            pred=[
                (it["main"]["transaction_id"] == it["transactions"]["transaction_id"]),
                (it["transactions"]["product_id"] == it["products"]["product_id"]),
            ]
        )
        map_1 = Map(
            columns={
                "[main](transaction_id)[transactions](product_id)[products]|price|identity": it[
                    "price"
                ],
                "[main](transaction_id)[transactions](product_id)[products]|type|identity": it[
                    "type"
                ],
            },
            remainder="drop",
        )
        pipeline_6 = join_1 >> map_1
        join_2 = Join(
            pred=[
                (it["main"]["transaction_id"] == it["transactions"]["transaction_id"])
            ]
        )
        map_2 = Map(
            columns={
                "[main](transaction_id)[transactions]|description|identity": it[
                    "description"
                ],
                "[main](transaction_id)[transactions]|product_id|identity": it[
                    "product_id"
                ],
            },
            remainder="drop",
        )
        pipeline_7 = join_2 >> map_2
        map_3 = Map(
            columns=[
                string_indexer(it["[main]|comments|identity"]),
                string_indexer(
                    it["[main](transaction_id)[transactions]|description|identity"]
                ),
                string_indexer(
                    it[
                        "[main](transaction_id)[transactions](product_id)[products]|type|identity"
                    ]
                ),
                string_indexer(
                    it["[main](group_customer_id)[customers]|name|identity"]
                ),
                string_indexer(
                    it["[main](group_customer_id)[customers]|address|identity"]
                ),
            ]
        )
        pipeline_8 = ConcatFeatures() >> map_3
        relational = Relational(
            operator=make_pipeline_graph(
                steps=[
                    scan,
                    scan_0,
                    pipeline_4,
                    scan_1,
                    pipeline_5,
                    map_0,
                    scan_2,
                    scan_3,
                    pipeline_6,
                    pipeline_7,
                    pipeline_8,
                ],
                edges=[
                    (scan, pipeline_4),
                    (scan, pipeline_5),
                    (scan, map_0),
                    (scan, pipeline_6),
                    (scan, pipeline_7),
                    (scan_0, pipeline_4),
                    (pipeline_4, pipeline_8),
                    (scan_1, pipeline_5),
                    (pipeline_5, pipeline_8),
                    (map_0, pipeline_8),
                    (scan_2, pipeline_6),
                    (scan_2, pipeline_7),
                    (scan_3, pipeline_6),
                    (pipeline_6, pipeline_8),
                    (pipeline_7, pipeline_8),
                ],
            )
        )
        pipeline = relational >> (KNeighborsClassifier | LogisticRegression)
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        from lale.lib.lale import Hyperopt

        opt = Hyperopt(estimator=pipeline, max_evals=2)
        opt.fit(X, y)


class TestRelationalOperator(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_fit_transform(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        trained_relational = relational.fit(self.X_train, self.y_train)
        _ = trained_relational.transform(self.X_test)

    def test_fit_error(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        with self.assertRaises(ValueError):
            _ = relational.fit([self.X_train], self.y_train)

    def test_transform_error(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        trained_relational = relational.fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            _ = trained_relational.transform([self.X_test])

    def test_fit_transform_in_pipeline(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        pipeline = relational >> LogisticRegression()
        trained_pipeline = pipeline.fit(self.X_train, self.y_train)
        _ = trained_pipeline.predict(self.X_test)


class TestMapSpark(unittest.TestCase):
    def setUp(self):
        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            self.sqlCtx = SQLContext(sc)

    def test_transform_spark_replace_list(self):
        if spark_installed:
            d = {
                "gender": ["m", "f", "m", "m", "f"],
                "state": ["NY", "NY", "CA", "NY", "CA"],
                "status": [0, 1, 1, 0, 1],
            }
            df = pd.DataFrame(data=d)
            sdf = self.sqlCtx.createDataFrame(df)
            gender_map = {"m": "Male", "f": "Female"}
            state_map = {"NY": "New York", "CA": "California"}
            trainable = Map(
                columns=[replace(it.gender, gender_map), replace(it.state, state_map)]
            )
            trained = trainable.fit(sdf)
            transformed_df = trained.transform(sdf)
            self.assertEqual(
                (transformed_df.count(), len(transformed_df.columns)), (5, 3)
            )
            self.assertEqual(transformed_df.head()[0], "Male")
            self.assertEqual(transformed_df.head()[1], "New York")

    def test_transform_spark_replace_map(self):
        if spark_installed:
            d = {
                "gender": ["m", "f", "m", "m", "f"],
                "state": ["NY", "NY", "CA", "NY", "CA"],
                "status": [0, 1, 1, 0, 1],
            }
            df = pd.DataFrame(data=d)
            sdf = self.sqlCtx.createDataFrame(df)
            gender_map = {"m": "Male", "f": "Female"}
            state_map = {"NY": "New York", "CA": "California"}
            trainable = Map(
                columns={
                    "new_gender": replace(it.gender, gender_map),
                    "new_state": replace(it.state, state_map),
                }
            )
            trained = trainable.fit(sdf)
            transformed_df = trained.transform(sdf)
            self.assertEqual(
                (transformed_df.count(), len(transformed_df.columns)), (5, 3)
            )
            self.assertEqual(transformed_df.head()[1], "Male")
            self.assertEqual(transformed_df.head()[2], "New York")

    def test_transform_dom_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-27", "2016-07-26"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[day_of_month(it.date_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 28)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 27)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 26)

    def test_transform_dom_fmt_list(self):
        df = pd.DataFrame({"date_column": ["28/05/2016", "27/06/2016", "26/07/2016"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[day_of_month(it.date_column, "d/M/y")])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 28)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 27)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 26)

    def test_transform_dom_fmt_map(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-27", "2016-07-26"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns={"dom": day_of_month(it.date_column, "y-M-d")})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)

        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["dom"], 28)
        self.assertEqual(transformed_df.collect()[1]["dom"], 27)
        self.assertEqual(transformed_df.collect()[2]["dom"], 26)

    def test_transform_dow_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-28", "2016-07-28"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[day_of_week(it.date_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        # Note that spark dayofweek outputs are different from pandas
        self.assertEqual(transformed_df.collect()[0]["date_column"], 7)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 3)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 5)

    def test_transform_dow_fmt_list(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-28", "2016-07-28"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[day_of_week(it.date_column, "y-M-d")])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 7)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 3)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 5)

    def test_transform_dow_fmt_map(self):
        df = pd.DataFrame({"date_column": ["2016-05-28", "2016-06-28", "2016-07-28"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns={"dow": day_of_week(it.date_column, "y-M-d")})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["dow"], 7)
        self.assertEqual(transformed_df.collect()[1]["dow"], 3)
        self.assertEqual(transformed_df.collect()[2]["dow"], 5)

    def test_transform_doy_list(self):
        df = pd.DataFrame({"date_column": ["2016-01-01", "2016-06-28", "2016-07-28"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[day_of_year(it.date_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 1)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 180)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 210)

    def test_transform_doy_fmt_list(self):
        df = pd.DataFrame({"date_column": ["2016-01-01", "2016-06-28", "2016-07-28"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[day_of_year(it.date_column, "y-M-d")])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 1)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 180)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 210)

    def test_transform_doy_fmt_map(self):
        df = pd.DataFrame({"date_column": ["2016-01-01", "2016-06-28", "2016-07-28"]})
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns={"doy": day_of_year(it.date_column, "y-M-d")})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["doy"], 1)
        self.assertEqual(transformed_df.collect()[1]["doy"], 180)
        self.assertEqual(transformed_df.collect()[2]["doy"], 210)

    def test_transform_hour_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[hour(it.date_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 15)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 12)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 1)

    def test_transform_hour_fmt_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[hour(it.date_column, "y-M-d HH:mm:ss")])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 15)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 12)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 1)

    def test_transform_hour_fmt_map(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns={"hour": hour(it.date_column, "y-M-d HH:mm:ss")})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["hour"], 15)
        self.assertEqual(transformed_df.collect()[1]["hour"], 12)
        self.assertEqual(transformed_df.collect()[2]["hour"], 1)

    def test_transform_minute_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[minute(it.date_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 16)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 18)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 1)

    def test_transform_minute_fmt_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns=[minute(it.date_column, "y-M-d HH:mm:ss")])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 16)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 18)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 1)

    def test_transform_minute_fmt_map(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns={"minute": minute(it.date_column, "y-M-d HH:mm:ss")})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["minute"], 16)
        self.assertEqual(transformed_df.collect()[1]["minute"], 18)
        self.assertEqual(transformed_df.collect()[2]["minute"], 1)

    def test_transform_month_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[month(it.date_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 1)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 6)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 7)

    def test_transform_month_fmt_list(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns=[month(it.date_column, "y-M-d HH:mm:ss")])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["date_column"], 1)
        self.assertEqual(transformed_df.collect()[1]["date_column"], 6)
        self.assertEqual(transformed_df.collect()[2]["date_column"], 7)

    def test_transform_month_fmt_map(self):
        df = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        sdf = self.sqlCtx.createDataFrame(df)

        trainable = Map(columns={"month": month(it.date_column, "y-M-d HH:mm:ss")})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["month"], 1)
        self.assertEqual(transformed_df.collect()[1]["month"], 6)
        self.assertEqual(transformed_df.collect()[2]["month"], 7)

    def test_transform_string_indexer_list(self):
        df = pd.DataFrame({"cat_column": ["a", "b", "b"]})
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns=[string_indexer(it.cat_column)])
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(transformed_df.collect()[0]["cat_column"], 1)
        self.assertEqual(transformed_df.collect()[1]["cat_column"], 0)
        self.assertEqual(transformed_df.collect()[2]["cat_column"], 0)

    def test_transform_string_indexer_map(self):
        df = pd.DataFrame({"cat_column": ["a", "b", "b"]})
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns={"string_indexed": string_indexer(it.cat_column)})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (3, 1))
        self.assertEqual(transformed_df.collect()[0]["string_indexed"], 1)
        self.assertEqual(transformed_df.collect()[1]["string_indexed"], 0)
        self.assertEqual(transformed_df.collect()[2]["string_indexed"], 0)
