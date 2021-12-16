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
import numpy as np
import pandas as pd

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row, SQLContext
    from pyspark.sql import functions as f

    spark_installed = True
except ImportError:
    spark_installed = False

from test import EnableSchemaValidation

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from lale.datasets.data_schemas import add_table_name, get_table_name
from lale.datasets.multitable import multitable_train_test_split
from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.expressions import (
    asc,
    collect_set,
    count,
    day_of_month,
    day_of_week,
    day_of_year,
    desc,
    first,
    hour,
    identity,
    isnan,
    isnotnan,
    isnotnull,
    isnull,
    it,
    max,
    mean,
    min,
    minute,
    month,
    replace,
    sum,
)
from lale.helpers import _is_pandas_df, _is_spark_df
from lale.lib.lale import (
    Alias,
    Filter,
    GroupBy,
    Hyperopt,
    Join,
    OrderBy,
    Relational,
    Scan,
    SplitXy,
)
from lale.lib.rasl import Aggregate, Map
from lale.lib.sklearn import PCA, LogisticRegression


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
        self.df1 = add_table_name(pd.DataFrame(data=table1), "main")

        table3 = {
            "tid": [2, 3, 4, 5],
            "col5": ["Warm", "Cold", "Warm", "Cold"],
        }
        self.df3 = add_table_name(pd.DataFrame(data=table3), "t1")

        table5 = {
            "TrainId": [1, 2, 3, 4, 5],
            "col3": ["NY", "NY", "CA", "TX", "TX"],
            "col4": [1, 1, 4, 8, 0],
            "col6": [1, np.nan, 2, None, 3],
        }
        self.df5 = add_table_name(pd.DataFrame(data=table5), "info")

        trainable = Join(
            pred=[
                it.main.train_id == it.info.TrainId,
                it.info.TrainId == it.t1.tid,
            ],
            join_type="left",
        )
        self.transformed_df = trainable.transform([self.df1, self.df5, self.df3])
        self.assertEqual(self.transformed_df.shape, (5, 9))
        self.assertEqual(self.transformed_df["col3"][2], "CA")

    def test_filter_pandas_isnan(self):
        trainable = Filter(pred=[isnan(it.col6)])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 9))
        self.assertTrue(np.all(np.isnan(filtered_df["col6"])))

    def test_filter_pandas_isnotnan(self):
        trainable = Filter(pred=[isnotnan(it.col6)])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 9))
        self.assertTrue(np.any(np.logical_not(np.isnan(filtered_df["col6"]))))

    def test_filter_pandas_isnull(self):
        trainable = Filter(pred=[isnull(it.col6)])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 9))
        self.assertTrue(np.all(np.isnan(filtered_df["col6"])))

    def test_filter_pandas_isnotnull(self):
        trainable = Filter(pred=[isnotnull(it.col6)])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 9))
        self.assertTrue(np.any(np.logical_not(np.isnan(filtered_df["col6"]))))

    def test_filter_pandas_eq(self):
        trainable = Filter(pred=[it.col3 == "TX"])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 9))
        self.assertTrue(np.all(filtered_df["col3"] == "TX"))

    def test_filter_pandas_neq(self):
        trainable = Filter(pred=[it.col1 != it["col3"]])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 9))
        self.assertTrue(np.all(filtered_df["col1"] != filtered_df["col3"]))

    def test_filter_pandas_ge(self):
        trainable = Filter(pred=[it["col4"] >= 5])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (1, 9))
        self.assertTrue(np.all(filtered_df["col4"] >= 5))

    def test_filter_pandas_gt(self):
        trainable = Filter(pred=[it["train_id"] > it.col4])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 9))
        self.assertTrue(np.all(filtered_df["train_id"] != filtered_df["col4"]))

    def test_filter_pandas_le(self):
        trainable = Filter(pred=[it["col3"] <= "NY"])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 9))
        self.assertTrue(np.all(filtered_df["col3"] <= "NY"))

    def test_filter_pandas_lt(self):
        trainable = Filter(pred=[it["col4"] < it["TrainId"]])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 9))
        self.assertTrue(np.all(filtered_df["col4"] < filtered_df["TrainId"]))

    def test_filter_pandas_multiple1(self):
        trainable = Filter(pred=[it.col3 == "TX", it["col4"] > 4])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (1, 9))
        self.assertTrue(np.all(filtered_df["col3"] == "TX"))
        self.assertTrue(np.all(filtered_df["col4"] > 4))

    def test_filter_pandas_multiple2(self):
        trainable = Filter(pred=[it.col5 != "Cold", it.train_id <= 4])
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (3, 9))
        self.assertTrue(np.all(filtered_df["col5"] != "Cold"))
        self.assertTrue(np.all(filtered_df["train_id"] <= 4))

    def test_filter_pandas_multiple3(self):
        trainable = Filter(
            pred=[
                it["train_id"] == it["TrainId"],
                it["col2"] != it.col4,
                it.col5 == "Cold",
            ]
        )
        filtered_df = trainable.transform(self.transformed_df)
        self.assertEqual(filtered_df.shape, (2, 9))
        self.assertTrue(np.all(filtered_df["train_id"] == filtered_df["train_id"]))
        self.assertTrue(np.all(filtered_df["col2"] != filtered_df["col4"]))
        self.assertTrue(np.all(filtered_df["col5"] == "Cold"))

    def test_filter_pandas_no_col_error(self):
        with self.assertRaises(ValueError):
            trainable = Filter(pred=[it["TrainId"] < it.col_na])
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
            self.spark_df2 = add_table_name(sqlContext.createDataFrame(table2), "info")

            l3 = [(2, "Warm"), (3, "Cold"), (4, "Warm"), (5, "Cold")]
            rdd = sc.parallelize(l3)
            table3 = rdd.map(lambda x: Row(tid=int(x[0]), col5=x[1]))
            self.spark_df3 = add_table_name(sqlContext.createDataFrame(table3), "t1")

            l4 = [
                (1, "NY", 1, float(1)),
                (2, "TX", 6, np.nan),
                (3, "CA", 2, float(2)),
                (4, "NY", 5, None),
                (5, "CA", 0, float(3)),
            ]
            rdd = sc.parallelize(l4)
            table4 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col1=x[1], col2=int(x[2]), col6=x[3])
            )
            self.spark_df4 = add_table_name(sqlContext.createDataFrame(table4), "main")

            trainable = Join(
                pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
                join_type="left",
            )
            self.transformed_df = trainable.transform(
                [self.spark_df4, self.spark_df2, self.spark_df3]
            ).sort("TrainId")
            self.assertEqual(self.transformed_df.count(), 5)
            self.assertEqual(len(self.transformed_df.columns), 8)
            self.assertEqual(self.transformed_df.collect()[2]["col1"], "CA")

    def test_filter_spark_isnan(self):
        if spark_installed:
            trainable = Filter(pred=[isnan(it["col6"])])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 1)
            self.assertEqual(len(filtered_df.columns), 8)
            rows = filtered_df.rdd.collect()
            row0 = rows[0]
            self.assertEqual(row0[1], "TX")
            self.assertTrue(np.isnan(row0[3]))

    def test_filter_spark_isnotnan(self):
        if spark_installed:
            trainable = Filter(pred=[isnotnan(it["col6"])])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 4)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col6")).first()[0]
            self.assertTrue(np.all((not np.isnan(i) for i in test_list)))

    def test_filter_spark_isnull(self):
        if spark_installed:
            trainable = Filter(pred=[isnull(it["col6"])])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 1)
            self.assertEqual(len(filtered_df.columns), 8)
            rows = filtered_df.rdd.collect()
            row0 = rows[0]
            self.assertEqual(row0[1], "NY")
            self.assertIsNone(row0[3])

    def test_filter_spark_isnotnull(self):
        if spark_installed:
            trainable = Filter(pred=[isnotnull(it["col6"])])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 4)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col6")).first()[0]
            self.assertTrue(np.all(i is not None for i in test_list))

    def test_filter_spark_eq(self):
        if spark_installed:
            trainable = Filter(pred=[it["col3"] == "NY"])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col3")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) == "NY"))

    def test_filter_spark_neq(self):
        if spark_installed:
            trainable = Filter(pred=[it.col1 != it["col3"]])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 3)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col1")).first()[0]
            test_list1 = filtered_df.select(f.collect_list("col3")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) != pd.Series(test_list1)))

    def test_filter_spark_ge(self):
        if spark_installed:
            trainable = Filter(pred=[it["col4"] >= 150])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 3)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col4")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) >= 150))

    def test_filter_spark_gt(self):
        if spark_installed:
            trainable = Filter(pred=[it["col2"] > it.tid])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col2")).first()[0]
            test_list1 = filtered_df.select(f.collect_list("tid")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) > pd.Series(test_list1)))

    def test_filter_spark_le(self):
        if spark_installed:
            trainable = Filter(pred=[it.col3 <= "NY"])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 3)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col3")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) <= "NY"))

    def test_filter_spark_lt(self):
        if spark_installed:
            trainable = Filter(pred=[it.col2 < it.TrainId])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col2")).first()[0]
            test_list1 = filtered_df.select(f.collect_list("TrainId")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) < pd.Series(test_list1)))

    def test_filter_spark_multiple1(self):
        if spark_installed:
            trainable = Filter(pred=[it["col3"] == "TX", it.col4 >= 150])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 1)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col3")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) == "TX"))
            test_list = filtered_df.select(f.collect_list("col4")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) >= 150))

    def test_filter_spark_multiple2(self):
        if spark_installed:
            trainable = Filter(pred=[it["col5"] != "Cold", it["TrainId"] <= 4])
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("col5")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) != "Cold"))
            test_list = filtered_df.select(f.collect_list("TrainId")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) <= 4))

    def test_filter_spark_multiple3(self):
        if spark_installed:
            trainable = Filter(
                pred=[
                    it["tid"] == it["TrainId"],
                    it.col2 > it.tid,
                    it["col5"] == "Warm",
                ]
            )
            filtered_df = trainable.transform(self.transformed_df)
            self.assertEqual(filtered_df.count(), 2)
            self.assertEqual(len(filtered_df.columns), 8)
            test_list = filtered_df.select(f.collect_list("tid")).first()[0]
            test_list1 = filtered_df.select(f.collect_list("TrainId")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) == pd.Series(test_list1)))
            test_list = filtered_df.select(f.collect_list("col2")).first()[0]
            test_list1 = filtered_df.select(f.collect_list("tid")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) > pd.Series(test_list1)))
            test_list = filtered_df.select(f.collect_list("col5")).first()[0]
            self.assertTrue(np.all(pd.Series(test_list) == "Warm"))

    def test_filter_spark_no_col_error(self):
        with self.assertRaises(ValueError):
            trainable = Filter(pred=[it["TrainId"] < it.col_na])
            _ = trainable.transform(self.transformed_df)


class TestScan(unittest.TestCase):
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()

    def test_attribute(self):
        with EnableSchemaValidation():
            trained = Scan(table=it.go_products)
            transformed = trained.transform(self.go_sales)
            self.assertEqual(get_table_name(transformed), "go_products")
            self.assertIs(self.go_sales[3], transformed)

    def test_subscript(self):
        with EnableSchemaValidation():
            trained = Scan(table=it["go_products"])
            transformed = trained.transform(self.go_sales)
            self.assertEqual(get_table_name(transformed), "go_products")
            self.assertIs(self.go_sales[3], transformed)

    def test_error1(self):
        with EnableSchemaValidation():
            trained = Scan(table=it.go_products)
            with self.assertRaisesRegex(ValueError, "invalid X"):
                _ = trained.transform(self.go_sales[3])

    def test_error2(self):
        trained = Scan(table=it.unknown_table)
        with self.assertRaisesRegex(ValueError, "could not find 'unknown_table'"):
            _ = trained.transform(self.go_sales)

    def test_error3(self):
        with self.assertRaisesRegex(ValueError, "expected `it.table_name` or"):
            _ = Scan(table=(it.go_products == 42))


# Testing alias operator for pandas and spark dataframes
class TestAlias(unittest.TestCase):
    # Get go_sales dataset in pandas and spark dataframes
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()
        self.go_sales_spark = fetch_go_sales_dataset("spark")

    def test_alias_pandas(self):
        trainable = Alias(name="test_alias")
        go_products = self.go_sales[3]
        self.assertEqual(get_table_name(go_products), "go_products")
        transformed_df = trainable.transform(go_products)
        self.assertEqual(get_table_name(transformed_df), "test_alias")
        self.assertTrue(_is_pandas_df(transformed_df))
        self.assertEqual(transformed_df.shape, (274, 8))

    def test_alias_spark(self):
        trainable = Alias(name="test_alias")
        go_products = self.go_sales_spark[3]
        self.assertEqual(get_table_name(go_products), "go_products")
        transformed_df = trainable.transform(go_products)
        self.assertEqual(get_table_name(transformed_df), "test_alias")
        self.assertTrue(_is_spark_df(transformed_df))
        self.assertEqual(transformed_df.count(), 274)
        self.assertEqual(len(transformed_df.columns), 8)

    def test_alias_name_error(self):
        with self.assertRaises(jsonschema.ValidationError):
            _ = Alias()
        with self.assertRaises(jsonschema.ValidationError):
            _ = Alias(name="")
        with self.assertRaises(jsonschema.ValidationError):
            _ = Alias(name="    ")

    def test_filter_name(self):
        go_products = self.go_sales[3]
        trained = Filter(pred=[it["Unit cost"] >= 10])
        transformed = trained.transform(go_products)
        self.assertEqual(get_table_name(transformed), "go_products")

    def test_map_name(self):
        go_products = self.go_sales[3]
        trained = Map(columns={"unit_cost": it["Unit cost"]})
        transformed = trained.transform(go_products)
        self.assertEqual(get_table_name(transformed), "go_products")

    def test_join_name(self):
        trained = Join(
            pred=[it.go_1k["Retailer code"] == it.go_retailers["Retailer code"]],
            name="joined_tables",
        )
        transformed = trained.transform(self.go_sales)
        self.assertEqual(get_table_name(transformed), "joined_tables")

    def test_groupby_name(self):
        go_products = self.go_sales[3]
        trained = GroupBy(by=[it["Product line"]])
        transformed = trained.transform(go_products)
        self.assertEqual(get_table_name(transformed), "go_products")

    def test_aggregate_name(self):
        go_daily_sales = self.go_sales[1]
        group_by = GroupBy(by=[it["Retailer code"]])
        aggregate = Aggregate(columns={"min_quantity": min(it.Quantity)})
        trained = group_by >> aggregate
        transformed = trained.transform(go_daily_sales)
        self.assertEqual(get_table_name(transformed), "go_daily_sales")


# Testing group_by operator for pandas and spark dataframes
class TestGroupBy(unittest.TestCase):
    # Get go_sales dataset in pandas and spark dataframes
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()
        self.go_sales_spark = fetch_go_sales_dataset("spark")

    def test_groupby_pandas(self):
        trainable = GroupBy(by=[it["Product line"]])
        go_products = self.go_sales[3]
        assert get_table_name(go_products) == "go_products"
        grouped_df = trainable.transform(go_products)
        self.assertEqual(grouped_df.ngroups, 5)

    def test_groupby_pandas1(self):
        trainable = GroupBy(by=[it["Product line"], it.Product])
        go_products = self.go_sales[3]
        assert get_table_name(go_products) == "go_products"
        grouped_df = trainable.transform(go_products)
        self.assertEqual(grouped_df.ngroups, 144)

    def test_groupby_spark(self):
        trainable = GroupBy(by=[it["Product line"], it.Product])
        go_products_spark = self.go_sales_spark[3]
        assert get_table_name(go_products_spark) == "go_products"
        _ = trainable.transform(go_products_spark)

    def test_groupby_pandas_no_col(self):
        trainable = GroupBy(by=[it["Product line"], it.Product.name])
        go_products = self.go_sales[3]
        assert get_table_name(go_products) == "go_products"
        with self.assertRaises(ValueError):
            _ = trainable.transform(go_products)


# Testing Aggregate operator for both pandas and Spark
class TestAggregate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark"]
        cls.tgt2datasets = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}

    def test_sales_not_grouped(self):
        pipeline = Scan(table=it.go_daily_sales) >> Aggregate(
            columns={
                "min_method_code": min(it["Order method code"]),
                "max_method_code": max(it["Order method code"]),
                "collect_set('Order method code')": collect_set(
                    it["Order method code"]
                ),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (1, 3), tgt)
            self.assertEqual(result.loc[0, "min_method_code"], 1, tgt)
            self.assertEqual(result.loc[0, "max_method_code"], 7, tgt)
            self.assertEqual(
                sorted(result.loc[0, "collect_set('Order method code')"]),
                [1, 2, 3, 4, 5, 6, 7],
                tgt,
            )

    def test_sales_onekey_grouped(self):
        pipeline = (
            Scan(table=it.go_daily_sales)
            >> GroupBy(by=[it["Retailer code"]])
            >> Aggregate(
                columns={
                    "retailer_code": it["Retailer code"],
                    "min_method_code": min(it["Order method code"]),
                    "max_method_code": max(it["Order method code"]),
                    "min_quantity": min(it["Quantity"]),
                    "method_codes": collect_set(it["Order method code"]),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (289, 5))
            row = result[result.retailer_code == 1201]
            self.assertEqual(row.loc[row.index[0], "retailer_code"], 1201, tgt)
            self.assertEqual(row.loc[row.index[0], "min_method_code"], 2, tgt)
            self.assertEqual(row.loc[row.index[0], "max_method_code"], 6, tgt)
            self.assertEqual(row.loc[row.index[0], "min_quantity"], 1, tgt)
            self.assertEqual(
                sorted(row.loc[row.index[0], "method_codes"]), [2, 3, 4, 5, 6], tgt
            )

    def test_products_onekey_grouped(self):
        pipeline = (
            Scan(table=it.go_products)
            >> GroupBy(by=[it["Product line"]])
            >> Aggregate(
                columns={
                    "line": first(it["Product line"]),
                    "mean_uc": mean(it["Unit cost"]),
                    "min_up": min(it["Unit price"]),
                    "count_pc": count(it["Product color"]),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (5, 4))
            row = result[result.line == "Camping Equipment"]
            self.assertEqual(row.loc[row.index[0], "line"], "Camping Equipment", tgt)
            self.assertAlmostEqual(row.loc[row.index[0], "mean_uc"], 89.0, 1, tgt)
            self.assertEqual(row.loc[row.index[0], "min_up"], 2.06, tgt)
            self.assertEqual(row.loc[row.index[0], "count_pc"], 41, tgt)

    def test_sales_twokeys_grouped(self):
        pipeline = (
            Scan(table=it.go_daily_sales)
            >> GroupBy(by=[it["Product number"], it["Retailer code"]])
            >> Aggregate(
                columns={
                    "product": it["Product number"],
                    "retailer": it["Retailer code"],
                    "mean_quantity": mean(it.Quantity),
                    "max_usp": max(it["Unit sale price"]),
                    "count_quantity": count(it.Quantity),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (5000, 5))
            row = result[(result["product"] == 70240) & (result["retailer"] == 1205)]
            self.assertEqual(row.loc[row.index[0], "product"], 70240, tgt)
            self.assertEqual(row.loc[row.index[0], "retailer"], 1205, tgt)
            self.assertAlmostEqual(
                row.loc[row.index[0], "mean_quantity"], 48.39, 2, tgt
            )
            self.assertEqual(row.loc[row.index[0], "max_usp"], 122.70, tgt)
            self.assertEqual(row.loc[row.index[0], "count_quantity"], 41, tgt)

    def test_products_twokeys_grouped(self):
        pipeline = (
            Scan(table=it.go_products)
            >> GroupBy(by=[it["Product line"], it["Product brand"]])
            >> Aggregate(
                columns={
                    "sum_uc": sum(it["Unit cost"]),
                    "max_uc": max(it["Unit cost"]),
                    "line": first(it["Product line"]),
                    "brand": first(it["Product brand"]),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (30, 4))
            row = result[
                (result.line == "Camping Equipment") & (result.brand == "Star")
            ]
            self.assertEqual(row.loc[row.index[0], "sum_uc"], 1968.19, tgt)
            self.assertEqual(row.loc[row.index[0], "max_uc"], 490.00, tgt)
            self.assertEqual(row.loc[row.index[0], "line"], "Camping Equipment", tgt)
            self.assertEqual(row.loc[row.index[0], "brand"], "Star", tgt)

    def test_error_unknown_column(self):
        pipeline = (
            Scan(table=it.go_daily_sales)
            >> GroupBy(by=[it["Product number"]])
            >> Aggregate(columns={"mean_quantity": mean(it["Quantity_1"])})
        )
        with self.assertRaises(KeyError):
            _ = pipeline.transform(self.tgt2datasets["pandas"])

    def test_error_columns_not_dict(self):
        pipeline = (
            Scan(table=it.go_daily_sales)
            >> GroupBy(by=[it["Product number"]])
            >> Aggregate(columns=[mean(it["Quantity_1"])])
        )
        with self.assertRaises(ValueError):
            _ = pipeline.transform(self.tgt2datasets["pandas"])

    def test_error_X_not_pandas_or_Spark(self):
        trainable = Aggregate(columns={"mean_quantity": mean(it["Quantity_1"])})
        with self.assertRaises(ValueError):
            _ = trainable.transform(pd.Series([1, 2, 3]))


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
        self.df1 = add_table_name(pd.DataFrame(data=table1), "main")

        table2 = {
            "TrainId": [1, 2, 3],
            "col3": ["USA", "USA", "UK"],
            "col4": [100, 100, 200],
        }
        self.df2 = add_table_name(pd.DataFrame(data=table2), "info")

        table3 = {
            "tid": [1, 2, 3],
            "col5": ["Warm", "Cold", "Warm"],
        }
        self.df3 = add_table_name(pd.DataFrame(data=table3), "t1")

        table4 = {
            "TrainId": [1, 2, 3, 4, 5],
            "col1": ["NY", "TX", "CA", "NY", "CA"],
            "col2": [0, 1, 1, 0, 1],
        }
        self.df4 = add_table_name(pd.DataFrame(data=table4), "main")

        table5 = {
            "TrainId": [1, 2, 3],
            "col3": ["NY", "NY", "CA"],
            "col4": [100, 100, 200],
        }
        self.df5 = add_table_name(pd.DataFrame(data=table5), "info")

        table6 = {
            "t_id": [2, 3],
            "col6": ["USA", "UK"],
        }
        self.df6 = add_table_name(pd.DataFrame(data=table6), "t2")

    # Multiple elements in predicate with different key column names
    def test_join_pandas_multiple_left(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="inner",
        )
        transformed_df = trainable.transform([self.df1, self.df2, self.df3])
        self.assertEqual(transformed_df.shape, (3, 8))
        self.assertEqual(transformed_df["col5"][1], "Cold")

    # Multiple elements in predicate with identical key columns names
    def test_join_pandas_multiple_left1(self):
        trainable = Join(
            pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="left",
        )
        transformed_df = trainable.transform([self.df4, self.df2, self.df3])
        self.assertEqual(transformed_df.shape, (5, 7))
        self.assertEqual(transformed_df["col3"][2], "UK")

    # Invert one of the join conditions as compared to the test case: test_join_pandas_multiple_left
    def test_join_pandas_multiple_right(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.t1.tid == it.info.TrainId],
            join_type="right",
        )
        transformed_df = trainable.transform([self.df1, self.df2, self.df3])
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
        transformed_df = trainable.transform([self.df1, self.df5, self.df3, self.df6])
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
        transformed_df = trainable.transform([self.df1, self.df5, self.df3, self.df6])
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
            _ = trainable.transform([self.df1, self.df5, self.df3, self.df6])

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
            _ = trainable.transform([self.df1, self.df5, self.df3, self.df6])

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
            _ = trainable.transform([self.df1, self.df5, self.df3, self.df6])

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
            _ = trainable.transform([self.df1, self.df5, self.df3, self.df6])

    # A table to be joined not present in input X
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
            _ = trainable.transform([self.df5, self.df3])

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
            self.spark_df1 = add_table_name(sqlContext.createDataFrame(table1), "main")

            l2 = [(1, "USA", 100), (2, "USA", 100), (3, "UK", 200)]
            rdd = sc.parallelize(l2)
            table2 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col3=x[1], col4=int(x[2]))
            )
            self.spark_df2 = add_table_name(sqlContext.createDataFrame(table2), "info")

            l3 = [(1, "Warm"), (2, "Cold"), (3, "Warm")]
            rdd = sc.parallelize(l3)
            table3 = rdd.map(lambda x: Row(tid=int(x[0]), col5=x[1]))
            self.spark_df3 = add_table_name(sqlContext.createDataFrame(table3), "t1")

            l4 = [(1, "NY", 0), (2, "TX", 1), (3, "CA", 1), (4, "NY", 0), (5, "CA", 1)]
            rdd = sc.parallelize(l4)
            table4 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col1=x[1], col2=int(x[2]))
            )
            self.spark_df4 = add_table_name(sqlContext.createDataFrame(table4), "main")

            l5 = [(1, "NY", 100), (2, "NY", 100), (3, "CA", 200)]
            rdd = sc.parallelize(l5)
            table5 = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col3=x[1], col4=int(x[2]))
            )
            self.spark_df5 = add_table_name(sqlContext.createDataFrame(table5), "info")

            l6 = [(2, "USA"), (3, "UK")]
            rdd = sc.parallelize(l6)
            table6 = rdd.map(lambda x: Row(t_id=int(x[0]), col3=x[1]))
            self.spark_df6 = add_table_name(sqlContext.createDataFrame(table6), "t2")

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
                [self.spark_df1, self.spark_df2, self.spark_df3]
            )
            self.assertEqual(transformed_df.count(), 3)
            self.assertEqual(len(transformed_df.columns), 8)
            self.assertEqual(
                transformed_df.sort(["train_id"]).collect()[0]["col5"], "Warm"
            )

    # Multiple elements in predicate with identical key columns names
    def test_join_spark_multiple_left1(self):
        if spark_installed:
            trainable = Join(
                pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
                join_type="left",
            )
            transformed_df = trainable.transform(
                [self.spark_df4, self.spark_df2, self.spark_df3]
            )
            self.assertEqual(transformed_df.count(), 5)
            self.assertEqual(len(transformed_df.columns), 7)
            self.assertEqual(
                transformed_df.sort(["TrainId"]).collect()[2]["col1"], "CA"
            )

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
                [self.spark_df1, self.spark_df2, self.spark_df3]
            )
            self.assertEqual(transformed_df.count(), 3)
            self.assertEqual(len(transformed_df.columns), 8)
            self.assertEqual(transformed_df.sort(["train_id"]).collect()[0]["col2"], 0)

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
                [self.spark_df1, self.spark_df5, self.spark_df3, self.spark_df6]
            )
            self.assertEqual(transformed_df.count(), 5)
            self.assertEqual(len(transformed_df.columns), 8)
            self.assertEqual(transformed_df.sort(["train_id"]).collect()[0]["col2"], 0)

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
                    [self.spark_df1, self.spark_df5, self.spark_df3, self.spark_df6]
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
                    [self.spark_df1, self.spark_df5, self.spark_df3, self.spark_df6]
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
                    [self.spark_df1, self.spark_df5, self.spark_df3, self.spark_df6]
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
                _ = trainable.transform([self.spark_df5, self.spark_df3])

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

    # The rename column functionality implemented as part of identity function for Map operator
    # does not support explicit identity calls for now.
    def test_transform_identity_map(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(
            columns={
                "new_gender": it.gender,
                "new_status": it["status"],
            }
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(df["gender"][0], transformed_df["new_gender"][0])
        self.assertEqual(df["status"][3], transformed_df["new_status"][3])
        self.assertEqual(len(transformed_df.columns), 2)

    def test_transform_identity_map_passthrough(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(
            columns={
                "new_gender": it.gender,
                "new_status": it["status"],
            },
            remainder="passthrough",
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(df["gender"][0], transformed_df["new_gender"][0])
        self.assertEqual(df["status"][3], transformed_df["new_status"][3])
        self.assertEqual(df["state"][3], transformed_df["state"][3])
        self.assertEqual(len(transformed_df.columns), 3)

    def test_transform_identity_map_error(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        with self.assertRaises(ValueError):
            trainable = Map(columns={"   ": it.gender})
            trained = trainable.fit(df)
            _ = trained.transform(df)
        with self.assertRaises(ValueError):
            trainable = Map(columns={"new_name": it["  "]})
            trained = trainable.fit(df)
            _ = trained.transform(df)
        with self.assertRaises(ValueError):
            trainable = Map(columns=[it.gender])
            trained = trainable.fit(df)
            _ = trained.transform(df)
        with self.assertRaises(ValueError):
            trainable = Map(columns=[it.dummy])
            trained = trainable.fit(df)
            _ = trained.transform(df)

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
            remainder="passthrough",
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 3))
        self.assertEqual(transformed_df["gender"][0], "Male")
        self.assertEqual(transformed_df["state"][0], "New York")
        self.assertEqual(transformed_df["status"][0], 0)

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
        self.assertEqual(transformed_df.shape, (5, 2))
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
        self.assertEqual(transformed_df.shape, (5, 2))
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

    # XXX Removed because it uses `string_indexer`
    # def test_with_hyperopt2(self):
    #     from lale.expressions import (
    #         count,
    #         it,
    #         max,
    #         mean,
    #         min,
    #         sum,
    #         variance,
    #     )

    #     wrap_imported_operators()
    #     scan = Scan(table=it["main"])
    #     scan_0 = Scan(table=it["customers"])
    #     join = Join(
    #         pred=[
    #             (
    #                 it["main"]["group_customer_id"]
    #                 == it["customers"]["group_customer_id"]
    #             )
    #         ]
    #     )
    #     map = Map(
    #         columns={
    #             "[main](group_customer_id)[customers]|number_children|identity": it[
    #                 "number_children"
    #             ],
    #             "[main](group_customer_id)[customers]|name|identity": it["name"],
    #             "[main](group_customer_id)[customers]|income|identity": it["income"],
    #             "[main](group_customer_id)[customers]|address|identity": it["address"],
    #             "[main](group_customer_id)[customers]|age|identity": it["age"],
    #         },
    #         remainder="drop",
    #     )
    #     pipeline_4 = join >> map
    #     scan_1 = Scan(table=it["purchase"])
    #     join_0 = Join(
    #         pred=[(it["main"]["group_id"] == it["purchase"]["group_id"])],
    #         join_limit=50.0,
    #     )
    #     aggregate = Aggregate(
    #         columns={
    #             "[main](group_id)[purchase]|price|variance": variance(it["price"]),
    #             "[main](group_id)[purchase]|time|sum": sum(it["time"]),
    #             "[main](group_id)[purchase]|time|mean": mean(it["time"]),
    #             "[main](group_id)[purchase]|time|min": min(it["time"]),
    #             "[main](group_id)[purchase]|price|sum": sum(it["price"]),
    #             "[main](group_id)[purchase]|price|count": count(it["price"]),
    #             "[main](group_id)[purchase]|price|mean": mean(it["price"]),
    #             "[main](group_id)[purchase]|price|min": min(it["price"]),
    #             "[main](group_id)[purchase]|price|max": max(it["price"]),
    #             "[main](group_id)[purchase]|time|max": max(it["time"]),
    #             "[main](group_id)[purchase]|time|variance": variance(it["time"]),
    #         },
    #         group_by=it["row_id"],
    #     )
    #     pipeline_5 = join_0 >> aggregate
    #     map_0 = Map(
    #         columns={
    #             "[main]|group_customer_id|identity": it["group_customer_id"],
    #             "[main]|transaction_id|identity": it["transaction_id"],
    #             "[main]|group_id|identity": it["group_id"],
    #             "[main]|comments|identity": it["comments"],
    #             "[main]|id|identity": it["id"],
    #             "prefix_0_id": it["prefix_0_id"],
    #             "next_purchase": it["next_purchase"],
    #             "[main]|time|identity": it["time"],
    #         },
    #         remainder="drop",
    #     )
    #     scan_2 = Scan(table=it["transactions"])
    #     scan_3 = Scan(table=it["products"])
    #     join_1 = Join(
    #         pred=[
    #             (it["main"]["transaction_id"] == it["transactions"]["transaction_id"]),
    #             (it["transactions"]["product_id"] == it["products"]["product_id"]),
    #         ]
    #     )
    #     map_1 = Map(
    #         columns={
    #             "[main](transaction_id)[transactions](product_id)[products]|price|identity": it[
    #                 "price"
    #             ],
    #             "[main](transaction_id)[transactions](product_id)[products]|type|identity": it[
    #                 "type"
    #             ],
    #         },
    #         remainder="drop",
    #     )
    #     pipeline_6 = join_1 >> map_1
    #     join_2 = Join(
    #         pred=[
    #             (it["main"]["transaction_id"] == it["transactions"]["transaction_id"])
    #         ]
    #     )
    #     map_2 = Map(
    #         columns={
    #             "[main](transaction_id)[transactions]|description|identity": it[
    #                 "description"
    #             ],
    #             "[main](transaction_id)[transactions]|product_id|identity": it[
    #                 "product_id"
    #             ],
    #         },
    #         remainder="drop",
    #     )
    #     pipeline_7 = join_2 >> map_2
    #     map_3 = Map(
    #         columns=[
    #             string_indexer(it["[main]|comments|identity"]),
    #             string_indexer(
    #                 it["[main](transaction_id)[transactions]|description|identity"]
    #             ),
    #             string_indexer(
    #                 it[
    #                     "[main](transaction_id)[transactions](product_id)[products]|type|identity"
    #                 ]
    #             ),
    #             string_indexer(
    #                 it["[main](group_customer_id)[customers]|name|identity"]
    #             ),
    #             string_indexer(
    #                 it["[main](group_customer_id)[customers]|address|identity"]
    #             ),
    #         ]
    #     )
    #     pipeline_8 = ConcatFeatures() >> map_3
    #     relational = Relational(
    #         operator=make_pipeline_graph(
    #             steps=[
    #                 scan,
    #                 scan_0,
    #                 pipeline_4,
    #                 scan_1,
    #                 pipeline_5,
    #                 map_0,
    #                 scan_2,
    #                 scan_3,
    #                 pipeline_6,
    #                 pipeline_7,
    #                 pipeline_8,
    #             ],
    #             edges=[
    #                 (scan, pipeline_4),
    #                 (scan, pipeline_5),
    #                 (scan, map_0),
    #                 (scan, pipeline_6),
    #                 (scan, pipeline_7),
    #                 (scan_0, pipeline_4),
    #                 (pipeline_4, pipeline_8),
    #                 (scan_1, pipeline_5),
    #                 (pipeline_5, pipeline_8),
    #                 (map_0, pipeline_8),
    #                 (scan_2, pipeline_6),
    #                 (scan_2, pipeline_7),
    #                 (scan_3, pipeline_6),
    #                 (pipeline_6, pipeline_8),
    #                 (pipeline_7, pipeline_8),
    #             ],
    #         )
    #     )
    #     pipeline = relational >> (KNeighborsClassifier | LogisticRegression)
    #     from sklearn.datasets import load_iris

    #     X, y = load_iris(return_X_y=True)
    #     from lale.lib.lale import Hyperopt

    #     opt = Hyperopt(estimator=pipeline, max_evals=2)
    #     opt.fit(X, y)

    def test_transform_ratio_map(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns={"ratio_h_w": it.height / it.weight})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 1))
        self.assertEqual(transformed_df["ratio_h_w"][0], 0.1)

    def test_transform_ratio_map_subscript(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns={"ratio_h_w": it["height"] / it.weight})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 1))
        self.assertEqual(transformed_df["ratio_h_w"][0], 0.1)

    def test_transform_ratio_map_list(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns=[it.height / it.weight])
        trained = trainable.fit(df)
        with self.assertRaises(ValueError):
            _ = trained.transform(df)

    def test_transform_subtract_map(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns={"subtract_h_w": it.height - it.weight})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 1))
        self.assertEqual(transformed_df["subtract_h_w"][0], -27)

    def test_transform_subtract_map_subscript(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns={"subtract_h_w": it["height"] - it.weight})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 1))
        self.assertEqual(transformed_df["subtract_h_w"][0], -27)

    def test_transform_subtract_map_list(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns=[it.height - it.weight])
        trained = trainable.fit(df)
        with self.assertRaises(ValueError):
            _ = trained.transform(df)

    def test_transform_binops(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(
            columns={
                "add_h_w": it["height"] + it.weight,
                "add_h_2": it["height"] + 2,
                "sub_h_w": it["height"] - it.weight,
                "sub_h_2": it["height"] - 2,
                "mul_h_w": it["height"] * it.weight,
                "mul_h_2": it["height"] * 2,
                "div_h_w": it["height"] / it.weight,
                "div_h_2": it["height"] / 2,
                "floor_div_h_w": it["height"] // it.weight,
                "floor_div_h_2": it["height"] // 2,
                "mod_h_w": it["height"] % it.weight,
                "mod_h_2": it["height"] % 2,
                "pow_h_w": it["height"] ** it.weight,
                "pow_h_2": it["height"] ** 2,
            }
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 14))
        self.assertEqual(
            transformed_df["add_h_w"][1], df["height"][1] + df["weight"][1]
        )
        self.assertEqual(transformed_df["add_h_2"][1], df["height"][1] + 2)
        self.assertEqual(
            transformed_df["sub_h_w"][1], df["height"][1] - df["weight"][1]
        )
        self.assertEqual(transformed_df["sub_h_2"][1], df["height"][1] - 2)
        self.assertEqual(
            transformed_df["mul_h_w"][1], df["height"][1] * df["weight"][1]
        )
        self.assertEqual(transformed_df["mul_h_2"][1], df["height"][1] * 2)
        self.assertEqual(
            transformed_df["div_h_w"][1], df["height"][1] / df["weight"][1]
        )
        self.assertEqual(transformed_df["div_h_2"][1], df["height"][1] / 2)
        self.assertEqual(
            transformed_df["floor_div_h_w"][1], df["height"][1] // df["weight"][1]
        )
        self.assertEqual(transformed_df["floor_div_h_2"][1], df["height"][1] // 2)
        self.assertEqual(
            transformed_df["mod_h_w"][1], df["height"][1] % df["weight"][1]
        )
        self.assertEqual(transformed_df["mod_h_2"][1], df["height"][1] % 2)
        self.assertEqual(
            transformed_df["pow_h_w"][1], df["height"][1] ** df["weight"][1]
        )
        self.assertEqual(transformed_df["pow_h_2"][1], df["height"][1] ** 2)

    def test_transform_arithmetic_expression(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        trainable = Map(columns={"expr": (it["height"] + it.weight * 10) / 2})
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df.shape, (5, 1))
        self.assertEqual(
            transformed_df["expr"][2], (df["height"][2] + df["weight"][2] * 10) / 2
        )

    def test_transform_nested_expressions(self):
        d = {
            "month": ["jan", "feb", "mar", "may", "aug"],
        }
        df = pd.DataFrame(data=d)
        month_map = {
            "jan": "2021-01-01",
            "feb": "2021-02-01",
            "mar": "2021-03-01",
            "arp": "2021-04-01",
            "may": "2021-05-01",
            "jun": "2021-06-01",
            "jul": "2021-07-01",
            "aug": "2021-08-01",
            "sep": "2021-09-01",
            "oct": "2021-10-01",
            "nov": "2021-11-01",
            "dec": "2021-12-01",
        }
        trainable = Map(
            columns={
                "date": replace(it.month, month_map),
                "month_id": month(replace(it.month, month_map), "%Y-%m-%d"),
                "next_month_id": identity(
                    month(replace(it.month, month_map), "%Y-%m-%d") % 12 + 1  # type: ignore
                ),
            }
        )
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        self.assertEqual(transformed_df["date"][0], "2021-01-01")
        self.assertEqual(transformed_df["date"][1], "2021-02-01")
        self.assertEqual(transformed_df["month_id"][2], 3)
        self.assertEqual(transformed_df["month_id"][3], 5)
        self.assertEqual(transformed_df["next_month_id"][0], 2)
        self.assertEqual(transformed_df["next_month_id"][3], 6)
        self.assertEqual(transformed_df["next_month_id"][4], 9)


class TestMapOnBothPandasAndSpark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark"]
        cls.tgt2datasets = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}

    def test_replace_unknown_identity(self):
        pipeline = Scan(table=it.go_products) >> Map(
            columns={
                "prod": it["Product number"],
                "line": replace(
                    it["Product line"],
                    {"Camping Equipment": "C", "Personal Accessories": "P"},
                ),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (274, 2))
            self.assertEqual(result.loc[0, "prod"], 1110, tgt)
            self.assertEqual(result.loc[0, "line"], "C", tgt)
            self.assertEqual(result.loc[117, "prod"], 101110, tgt)
            self.assertEqual(result.loc[117, "line"], "Golf Equipment", tgt)
            self.assertEqual(result.loc[273, "prod"], 154150, tgt)
            self.assertEqual(result.loc[273, "line"], "P", tgt)

    def test_replace_unknown_encoded(self):
        pipeline = Scan(table=it.go_products) >> Map(
            columns={
                "prod": it["Product number"],
                "line": replace(
                    it["Product line"],
                    {"Camping Equipment": "C", "Personal Accessories": "P"},
                    handle_unknown="use_encoded_value",
                    unknown_value="U",
                ),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt == "spark":
                result = result.toPandas()
            self.assertEqual(result.shape, (274, 2))
            self.assertEqual(result.loc[0, "prod"], 1110, tgt)
            self.assertEqual(result.loc[0, "line"], "C", tgt)
            self.assertEqual(result.loc[117, "prod"], 101110, tgt)
            self.assertEqual(result.loc[117, "line"], "U", tgt)
            self.assertEqual(result.loc[273, "prod"], 154150, tgt)
            self.assertEqual(result.loc[273, "line"], "P", tgt)


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

    # The rename column functionality implemented as part of identity function for Map operator
    # does not support explicit identity calls for now.
    def test_transform_identity_map(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(
            columns={
                "new_gender": it.gender,
                "new_status": it["status"],
            }
        )
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(df["gender"][0], transformed_df.collect()[0]["new_gender"])
        self.assertEqual(df["status"][3], transformed_df.collect()[3]["new_status"])
        self.assertEqual(len(transformed_df.columns), 2)

    def test_transform_identity_map_passthrough(self):
        d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(
            columns={
                "new_gender": it.gender,
                "new_status": it["status"],
            },
            remainder="passthrough",
        )
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual(df["gender"][0], transformed_df.collect()[0]["new_gender"])
        self.assertEqual(df["status"][3], transformed_df.collect()[3]["new_status"])
        self.assertEqual(df["state"][2], transformed_df.collect()[2]["state"])
        self.assertEqual(len(transformed_df.columns), 3)

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
                (transformed_df.count(), len(transformed_df.columns)), (5, 2)
            )
            self.assertEqual(transformed_df.collect()[0]["gender"], "Male")
            self.assertEqual(transformed_df.collect()[0]["state"], "New York")

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
                (transformed_df.count(), len(transformed_df.columns)), (5, 2)
            )
            self.assertEqual(transformed_df.collect()[0]["new_gender"], "Male")
            self.assertEqual(transformed_df.collect()[0]["new_state"], "New York")

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

    def test_transform_ratio_map(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns={"ratio_h_w": it.height / it.weight})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (5, 1))
        self.assertEqual(transformed_df.collect()[0]["ratio_h_w"], 0.1)

    def test_transform_ratio_map_subscript(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns={"ratio_h_w": it["height"] / it.weight})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (5, 1))
        self.assertEqual(transformed_df.collect()[0]["ratio_h_w"], 0.1)

    def test_transform_ratio_map_list(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns=[it.height / it.weight])
        trained = trainable.fit(sdf)
        with self.assertRaises(ValueError):
            _ = trained.transform(df)

    def test_transform_subtract_map(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns={"subtraction_h_w": it.height - it.weight})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (5, 1))
        self.assertEqual(transformed_df.collect()[0]["subtraction_h_w"], -27)

    def test_transform_subtract_map_subscript(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns={"subtraction_h_w": it["height"] - it.weight})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (5, 1))
        self.assertEqual(transformed_df.collect()[0]["subtraction_h_w"], -27)

    def test_transform_subtract_map_list(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns=[it.height - it.weight])
        trained = trainable.fit(sdf)
        with self.assertRaises(ValueError):
            _ = trained.transform(df)

    def test_transform_binops(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(
            columns={
                "add_h_w": it["height"] + it.weight,
                "add_h_2": it["height"] + 2,
                "sub_h_w": it["height"] - it.weight,
                "sub_h_2": it["height"] - 2,
                "mul_h_w": it["height"] * it.weight,
                "mul_h_2": it["height"] * 2,
                "div_h_w": it["height"] / it.weight,
                "div_h_2": it["height"] / 2,
                "floor_div_h_w": it["height"] // it.weight,
                "floor_div_h_2": it["height"] // 2,
                "mod_h_w": it["height"] % it.weight,
                "mod_h_2": it["height"] % 2,
                "pow_h_w": it["height"] ** it.weight,
                "pow_h_2": it["height"] ** 2,
            }
        )
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (5, 14))
        transformed_df = transformed_df.toPandas()
        self.assertEqual(
            transformed_df["add_h_w"][1], df["height"][1] + df["weight"][1]
        )
        self.assertEqual(transformed_df["add_h_2"][1], df["height"][1] + 2)
        self.assertEqual(
            transformed_df["sub_h_w"][1], df["height"][1] - df["weight"][1]
        )
        self.assertEqual(transformed_df["sub_h_2"][1], df["height"][1] - 2)
        self.assertEqual(
            transformed_df["mul_h_w"][1], df["height"][1] * df["weight"][1]
        )
        self.assertEqual(transformed_df["mul_h_2"][1], df["height"][1] * 2)
        self.assertEqual(
            transformed_df["div_h_w"][1], df["height"][1] / df["weight"][1]
        )
        self.assertEqual(transformed_df["div_h_2"][1], df["height"][1] / 2)
        self.assertEqual(
            transformed_df["floor_div_h_w"][1], df["height"][1] // df["weight"][1]
        )
        self.assertEqual(transformed_df["floor_div_h_2"][1], df["height"][1] // 2)
        self.assertEqual(
            transformed_df["mod_h_w"][1], df["height"][1] % df["weight"][1]
        )
        self.assertEqual(transformed_df["mod_h_2"][1], df["height"][1] % 2)
        # Spark and Python have a different semantics
        # self.assertEqual(transformed_df["pow_h_w"][1], df["height"][1] ** df["weight"][1])
        self.assertEqual(transformed_df["pow_h_w"][1], 4 ** 50)
        self.assertEqual(transformed_df["pow_h_2"][1], df["height"][1] ** 2)

    def test_transform_arithmetic_expression(self):
        d = {
            "height": [3, 4, 6, 3, 5],
            "weight": [30, 50, 170, 40, 130],
            "status": [0, 1, 1, 0, 1],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        trainable = Map(columns={"expr": (it["height"] + it.weight * 10) / 2})
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        self.assertEqual((transformed_df.count(), len(transformed_df.columns)), (5, 1))
        transformed_df = transformed_df.collect()
        self.assertEqual(
            transformed_df[2]["expr"], (df["height"][2] + df["weight"][2] * 10) / 2
        )

    def test_transform_nested_expressions(self):
        d = {
            "month": ["jan", "feb", "mar", "may", "aug"],
        }
        df = pd.DataFrame(data=d)
        sdf = self.sqlCtx.createDataFrame(df)
        month_map = {
            "jan": "2021-01-01",
            "feb": "2021-02-01",
            "mar": "2021-03-01",
            "arp": "2021-04-01",
            "may": "2021-05-01",
            "jun": "2021-06-01",
            "jul": "2021-07-01",
            "aug": "2021-08-01",
            "sep": "2021-09-01",
            "oct": "2021-10-01",
            "nov": "2021-11-01",
            "dec": "2021-12-01",
        }
        trainable = Map(
            columns={
                "date": replace(it.month, month_map),
                "month_id": month(replace(it.month, month_map)),
                "next_month_id": identity(
                    month(replace(it.month, month_map)) % 12 + 1  # type: ignore
                ),
            }
        )  # type: ignore
        trained = trainable.fit(sdf)
        transformed_df = trained.transform(sdf)
        transformed_df = transformed_df.collect()
        self.assertEqual(transformed_df[0]["date"], "2021-01-01")
        self.assertEqual(transformed_df[1]["date"], "2021-02-01")
        self.assertEqual(transformed_df[2]["month_id"], 3)
        self.assertEqual(transformed_df[3]["month_id"], 5)
        self.assertEqual(transformed_df[0]["next_month_id"], 2)
        self.assertEqual(transformed_df[3]["next_month_id"], 6)
        self.assertEqual(transformed_df[4]["next_month_id"], 9)


class TestOrderBy(unittest.TestCase):
    def setUp(self):
        self.d = {
            "gender": ["m", "f", "m", "m", "f"],
            "state": ["NY", "NY", "CA", "NY", "CA"],
            "status": [0, 1, 1, 0, 1],
        }

    def test_order_attr1(self):
        df = pd.DataFrame(data=self.d)
        trainable = OrderBy(by=it.gender)
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)

        self.assertTrue((transformed_df["gender"]).is_monotonic_increasing)

    def test_order_attr1_asc(self):
        df = pd.DataFrame(data=self.d)
        trainable = OrderBy(by=asc(it.gender))
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)

        self.assertTrue((transformed_df["gender"]).is_monotonic_increasing)

    def test_order_attr1_desc(self):
        df = pd.DataFrame(data=self.d)
        trainable = OrderBy(by=desc(it.gender))
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)

        self.assertTrue((transformed_df["gender"]).is_monotonic_decreasing)

    def test_order_str1_desc(self):
        df = pd.DataFrame(data=self.d)
        trainable = OrderBy(by=desc("gender"))
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)

        self.assertTrue((transformed_df["gender"]).is_monotonic_decreasing)

    def test_order_multiple(self):
        df = pd.DataFrame(data=self.d)
        trainable = OrderBy(by=[it.gender, desc(it.status)])
        trained = trainable.fit(df)
        transformed_df = trained.transform(df)
        expected_result = pd.DataFrame(
            data={
                "gender": ["f", "f", "m", "m", "m"],
                "state": ["NY", "CA", "CA", "NY", "NY"],
                "status": [1, 1, 1, 0, 0],
            }
        )
        self.assertEqual(list(transformed_df.index), [1, 4, 2, 0, 3])
        self.assertTrue((transformed_df["gender"]).is_monotonic_increasing)
        self.assertTrue(transformed_df.reset_index(drop=True).equals(expected_result))


class TestOrderBySpark(unittest.TestCase):
    def setUp(self):
        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            self.sqlCtx = SQLContext(sc)
            d = {
                "gender": ["m", "f", "m", "m", "f"],
                "state": ["NY", "NY", "CA", "NY", "CA"],
                "status": [0, 1, 1, 0, 1],
            }
            df = pd.DataFrame(data=d)
            self.sdf = self.sqlCtx.createDataFrame(df)

    # The rename column functionality implemented as part of identity function for Map operator
    # does not support explicit identity calls for now.
    def test_str1(self):
        trainable = OrderBy(by="gender")
        trained = trainable.fit(self.sdf)
        transformed_df = trained.transform(self.sdf)
        cgender = pd.Series(
            transformed_df.select("gender").rdd.flatMap(lambda x: x).collect()
        )

        self.assertTrue(cgender.is_monotonic_increasing)

    def test_order_attr1(self):
        trainable = OrderBy(by=it.gender)
        trained = trainable.fit(self.sdf)
        transformed_df = trained.transform(self.sdf)

        cgender = pd.Series(
            transformed_df.select("gender").rdd.flatMap(lambda x: x).collect()
        )

        self.assertTrue(cgender.is_monotonic_increasing)

    def test_order_attr1_asc(self):
        trainable = OrderBy(by=asc(it.gender))
        trained = trainable.fit(self.sdf)
        transformed_df = trained.transform(self.sdf)

        cgender = pd.Series(
            transformed_df.select("gender").rdd.flatMap(lambda x: x).collect()
        )

        self.assertTrue(cgender.is_monotonic_increasing)

    def test_order_attr1_desc(self):
        trainable = OrderBy(by=desc(it.gender))
        trained = trainable.fit(self.sdf)
        transformed_df = trained.transform(self.sdf)

        cgender = pd.Series(
            transformed_df.select("gender").rdd.flatMap(lambda x: x).collect()
        )

        self.assertTrue(cgender.is_monotonic_decreasing)

    def test_order_str1_desc(self):
        trainable = OrderBy(by=desc("gender"))
        trained = trainable.fit(self.sdf)
        transformed_df = trained.transform(self.sdf)

        cgender = pd.Series(
            transformed_df.select("gender").rdd.flatMap(lambda x: x).collect()
        )

        self.assertTrue(cgender.is_monotonic_decreasing)

    def test_order_multiple(self):
        trainable = OrderBy(by=[it.gender, desc(it.status)])
        trained = trainable.fit(self.sdf)
        transformed_df = trained.transform(self.sdf)
        expected_result = pd.DataFrame(
            data={
                "gender": ["f", "f", "m", "m", "m"],
                "state": ["NY", "CA", "CA", "NY", "NY"],
                "status": [1, 1, 1, 0, 0],
            }
        )
        cgender = pd.Series(
            transformed_df.select("gender").rdd.flatMap(lambda x: x).collect()
        )
        self.assertTrue(cgender.is_monotonic_increasing)
        self.assertTrue(
            transformed_df.toPandas().reset_index(drop=True).equals(expected_result)
        )


class TestSplitXy(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            pd.DataFrame(X), pd.DataFrame(y)
        )
        self.combined_df = pd.concat([self.X_train, self.y_train], axis=1)
        self.combined_df.columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]

    def test_split_transform(self):
        pipeline = PCA()
        trainable = SplitXy(operator=pipeline, label_name="class")
        trained = trainable.fit(self.combined_df)
        _ = trained.transform(self.combined_df)

    def test_split_predict(self):
        pipeline = PCA() >> LogisticRegression(random_state=42)
        trainable = SplitXy(operator=pipeline, label_name="class")
        trained = trainable.fit(self.combined_df)
        _ = trained.predict(self.X_test)


@unittest.skip(
    "skipping because we don't have any estimators that handle a Spark dataframe."
)
class TestSplitXySpark(unittest.TestCase):
    def setUp(self):
        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            self.sqlCtx = SQLContext(sc)
            data = load_iris()
            X, y = data.data, data.target
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                pd.DataFrame(X), pd.DataFrame(y)
            )
            self.combined_df = pd.concat([self.X_train, self.y_train], axis=1)
            self.combined_df.columns = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "class",
            ]
            self.combined_df_spark = self.sqlCtx.createDataFrame(self.combined_df)

    def test_split_spark_transform(self):
        pipeline = PCA()
        trainable = SplitXy(operator=pipeline, label_name="class")
        trained = trainable.fit(self.combined_df_spark)
        _ = trained.transform(self.combined_df_spark)

    def test_split_spark_predict(self):
        pipeline = PCA() >> LogisticRegression(random_state=42)
        trainable = SplitXy(operator=pipeline, label_name="class")
        trained = trainable.fit(self.combined_df)
        _ = trained.predict(pd.DataFrame(self.X_test))


class TestTrainTestSplit(unittest.TestCase):
    # Get go_sales dataset in pandas and spark dataframes
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()
        self.go_sales_spark = fetch_go_sales_dataset("spark")

    def test_split_pandas(self):
        train, test, train_y, test_y = multitable_train_test_split(
            self.go_sales,
            main_table_name="go_products",
            label_column_name="Product number",
            test_size=0.2,
        )
        main_table_df: pd.Dataframe = None
        for df in train:
            if get_table_name(df) == "go_products":
                main_table_df = df
        self.assertEqual(len(main_table_df), 220)
        self.assertEqual(len(train_y), 220)
        for df in test:
            if get_table_name(df) == "go_products":
                main_table_df = df
        self.assertEqual(len(main_table_df), 54)
        self.assertEqual(len(test_y), 54)

    def test_split_pandas_1(self):
        train, test, train_y, test_y = multitable_train_test_split(
            self.go_sales,
            main_table_name="go_products",
            label_column_name="Product number",
            test_size=200,
        )
        main_table_df: pd.Dataframe = None
        for df in test:
            if get_table_name(df) == "go_products":
                main_table_df = df
        self.assertEqual(len(main_table_df), 200)
        self.assertEqual(len(test_y), 200)

    def test_split_spark(self):
        train, test, train_y, test_y = multitable_train_test_split(
            self.go_sales_spark,
            main_table_name="go_products",
            label_column_name="Product number",
            test_size=0.2,
        )
        main_table_df: pd.Dataframe = None
        for df in train:
            if get_table_name(df) == "go_products":
                main_table_df = df
        self.assertEqual(main_table_df.count(), 220)
        self.assertEqual(train_y.count(), 220)
        for df in test:
            if get_table_name(df) == "go_products":
                main_table_df = df
        self.assertEqual(main_table_df.count(), 54)
        self.assertEqual(test_y.count(), 54)
