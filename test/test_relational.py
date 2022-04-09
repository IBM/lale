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

import lale.operators
from lale.lib.rasl.convert import Convert
from lale.operator_wrapper import wrap_imported_operators

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row, SQLContext

    from lale.datasets.data_schemas import SparkDataFrameWithIndex

    spark_installed = True
except ImportError:
    spark_installed = False

from test import EnableSchemaValidation

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from lale.datasets import pandas2spark
from lale.datasets.data_schemas import add_table_name, get_index_name, get_table_name
from lale.datasets.multitable import multitable_train_test_split
from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.expressions import (
    asc,
    astype,
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
    median,
    min,
    minute,
    mode,
    month,
    replace,
    string_indexer,
    sum,
)
from lale.helpers import (
    _ensure_pandas,
    _is_pandas_df,
    _is_spark_df,
    _is_spark_with_index,
)
from lale.lib.dataframe import get_columns
from lale.lib.lale import ConcatFeatures, Hyperopt, SplitXy
from lale.lib.rasl import (
    Aggregate,
    Alias,
    Filter,
    GroupBy,
    Join,
    Map,
    OrderBy,
    Relational,
    Scan,
)
from lale.lib.sklearn import PCA, KNeighborsClassifier, LogisticRegression


def _set_index_name(df, name):
    return add_table_name(df.rename_axis(index=name), get_table_name(df))


def _set_index(df, name):
    return add_table_name(df.set_index(name), get_table_name(df))


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


# Testing filter operator
class TestFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        info = [
            (1, "NY", 100),
            (2, "NY", 150),
            (3, "TX", 200),
            (4, "TX", 100),
            (5, "CA", 200),
        ]
        t1 = [(2, "Warm"), (3, "Cold"), (4, "Warm"), (5, "Cold")]
        main = [
            (1, "NY", 1, float(1)),
            (2, "TX", 6, np.nan),
            (3, "CA", 2, float(2)),
            (4, "NY", 5, None),
            (5, "CA", 0, float(3)),
        ]

        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            sqlContext = SQLContext(sc)

            rdd = sc.parallelize(main)
            table_main = rdd.map(
                lambda x: Row(TrainId=int(x[0]), col1=x[1], col2=int(x[2]), col6=x[3])
            )
            spark_main = add_table_name(sqlContext.createDataFrame(table_main), "main")

            rdd = sc.parallelize(info)
            table_info = rdd.map(
                lambda x: Row(train_id=int(x[0]), col3=x[1], col4=int(x[2]))
            )
            spark_info = add_table_name(sqlContext.createDataFrame(table_info), "info")

            rdd = sc.parallelize(t1)
            table_t1 = rdd.map(lambda x: Row(tid=int(x[0]), col5=x[1]))
            spark_t1 = add_table_name(sqlContext.createDataFrame(table_t1), "t1")

            trainable = Join(
                pred=[
                    it.main.TrainId == it.info.train_id,
                    it.info.train_id == it.t1.tid,
                ],
                join_type="left",
            )
            spark_transformed_df = trainable.transform(
                [spark_main, spark_info, spark_t1]
            ).sort("TrainId")
            cls.tgt2datasets = {
                "pandas": spark_transformed_df.toPandas(),
                "spark": spark_transformed_df,
                "spark-with-index": SparkDataFrameWithIndex(spark_transformed_df),
            }
        else:
            pandas_main = pd.DataFrame(main, index=["TrainId", "col1", "col2", "col6"])
            pandas_info = pd.DataFrame(info, index=["train_id", "col3", "col4"])
            pandas_t1 = pd.DataFrame(t1, index=["tid", "col5"])
            trainable = Join(
                pred=[
                    it.main.TrainId == it.info.train_id,
                    it.info.train_id == it.t1.tid,
                ],
                join_type="left",
            )
            pandas_transformed_df = trainable.transform(
                [pandas_main, pandas_info, pandas_t1]
            ).sort("TrainId")
            cls.tgt2datasets = {"pandas": pandas_transformed_df}

    def test_filter_isnan(self):
        pandas_transformed_df = self.tgt2datasets["pandas"]
        self.assertEqual(pandas_transformed_df.shape, (5, 9))
        self.assertEqual(pandas_transformed_df["col1"][2], "CA")
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[isnan(it.col6)])
            filtered_df = trainable.transform(transformed_df)
            if tgt == "pandas":
                # `None` is considered as `nan` in Pandas
                self.assertEqual(filtered_df.shape, (2, 9), tgt)
                self.assertTrue(all(np.isnan(filtered_df["col6"])), tgt)
            elif tgt.startswith("spark"):
                self.assertEqual(_ensure_pandas(filtered_df).shape, (1, 9), tgt)
                test_list = [row[0] for row in filtered_df.select("col6").collect()]
                self.assertTrue(all((np.isnan(i) for i in test_list if i is not None)))
            else:
                assert False
            if tgt == "spark-with-index":
                self.assertEqual(
                    get_index_name(transformed_df), get_index_name(filtered_df)
                )

    def test_filter_isnotnan(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[isnotnan(it.col6)])
            filtered_df = trainable.transform(transformed_df)
            if tgt == "pandas":
                self.assertTrue(all(np.logical_not(np.isnan(filtered_df["col6"]))), tgt)
                self.assertEqual(filtered_df.shape, (3, 9), tgt)
            elif tgt.startswith("spark"):
                self.assertEqual(_ensure_pandas(filtered_df).shape, (4, 9), tgt)
                test_list = [row[0] for row in filtered_df.select("col6").collect()]
                self.assertTrue(
                    all((not np.isnan(i) for i in test_list if i is not None))
                )
            else:
                assert False
            if tgt == "spark-with-index":
                self.assertEqual(
                    get_index_name(transformed_df), get_index_name(filtered_df)
                )

    def test_filter_isnull(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[isnull(it.col6)])
            filtered_df = trainable.transform(transformed_df)
            if tgt == "pandas":
                # `None` is considered as `nan` in Pandas
                self.assertEqual(filtered_df.shape, (2, 9), tgt)
                self.assertTrue(all(np.isnan(filtered_df["col6"])), tgt)
            elif tgt.startswith("spark"):
                self.assertEqual(_ensure_pandas(filtered_df).shape, (1, 9), tgt)
                test_list = [row[0] for row in filtered_df.select("col6").collect()]
                self.assertTrue(all((i is None for i in test_list)))
            else:
                assert False
            if tgt == "spark-with-index":
                self.assertEqual(
                    get_index_name(transformed_df), get_index_name(filtered_df)
                )

    def test_filter_isnotnull(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[isnotnull(it.col6)])
            filtered_df = trainable.transform(transformed_df)
            if tgt == "pandas":
                # `None` is considered as `nan` in Pandas
                self.assertEqual(filtered_df.shape, (3, 9), tgt)
                self.assertTrue(all(np.logical_not(np.isnan(filtered_df["col6"]))))
            elif tgt.startswith("spark"):
                self.assertEqual(_ensure_pandas(filtered_df).shape, (4, 9), tgt)
                test_list = [row[0] for row in filtered_df.select("col6").collect()]
                self.assertTrue(all((i is not None for i in test_list)))
            else:
                assert False
            if tgt == "spark-with-index":
                self.assertEqual(
                    get_index_name(transformed_df), get_index_name(filtered_df)
                )

    def test_filter_eq(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it.col3 == "TX"])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (2, 9), tgt)
            self.assertTrue(all(filtered_df["col3"] == "TX"), tgt)

    def test_filter_neq(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it.col1 != it["col3"]])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (3, 9), tgt)
            self.assertTrue(all(filtered_df["col1"] != filtered_df["col3"]), tgt)

    def test_filter_ge(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it["col4"] >= 150])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (3, 9), tgt)
            self.assertTrue(all(filtered_df["col4"] >= 150), tgt)

    def test_filter_gt(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it["col4"] > 150])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (2, 9), tgt)
            self.assertTrue(all(filtered_df["col4"] > 150), tgt)

    def test_filter_le(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it["col3"] <= "NY"])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (3, 9), tgt)
            self.assertTrue(all(filtered_df["col3"] <= "NY"), tgt)

    def test_filter_lt(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it["col2"] < it["TrainId"]])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (2, 9), tgt)
            self.assertTrue(all(filtered_df["col2"] < filtered_df["TrainId"]), tgt)

    def test_filter_multiple1(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it.col3 == "TX", it["col2"] > 4])
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (1, 9))
            self.assertTrue(all(filtered_df["col3"] == "TX"), tgt)
            self.assertTrue(all(filtered_df["col2"] > 4), tgt)

    def test_filter_multiple2(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(pred=[it.col5 != "Cold", it.train_id < 4])
            filtered_df = trainable.transform(transformed_df)
            if tgt == "pandas":
                self.assertEqual(filtered_df.shape, (2, 9))
            elif tgt.startswith("spark"):
                # `None != "Cold"` is not true in Spark
                filtered_df = _ensure_pandas(filtered_df)
                self.assertEqual(filtered_df.shape, (1, 9))
            else:
                assert False
            self.assertTrue(all(filtered_df["col5"] != "Cold"), tgt)
            self.assertTrue(all(filtered_df["train_id"] < 4), tgt)

    def test_multiple3(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            trainable = Filter(
                pred=[
                    it["tid"] == it["TrainId"],
                    it["col2"] >= it.train_id,
                    it.col3 == "NY",
                ]
            )
            filtered_df = trainable.transform(transformed_df)
            filtered_df = _ensure_pandas(filtered_df)
            self.assertEqual(filtered_df.shape, (1, 9), tgt)
            self.assertTrue(all(filtered_df["tid"] == filtered_df["TrainId"]), tgt)
            self.assertTrue(all(filtered_df["col2"] >= filtered_df["train_id"]), tgt)
            self.assertTrue(all(filtered_df["col3"] == "NY"), tgt)

    def test_filter_no_col_error(self):
        for tgt, transformed_df in self.tgt2datasets.items():
            with self.assertRaises(ValueError):
                trainable = Filter(pred=[it["TrainId"] < it.col_na])
                _ = trainable.transform(transformed_df)


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
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}

    def test_alias(self):
        for tgt, datasets in self.tgt2datasets.items():
            trainable = Alias(name="test_alias")
            go_products = datasets[3]
            self.assertEqual(get_table_name(go_products), "go_products")
            transformed_df = trainable.transform(go_products)
            self.assertEqual(get_table_name(transformed_df), "test_alias")
            if tgt == "pandas":
                self.assertTrue(_is_pandas_df(transformed_df))
            elif tgt.startswith("spark"):
                self.assertTrue(_is_spark_df(transformed_df))
                transformed_df = transformed_df.toPandas()
            else:
                assert False
            self.assertEqual(transformed_df.shape, (274, 8))

    def test_alias_name_error(self):
        with self.assertRaises(jsonschema.ValidationError):
            _ = Alias()
        with self.assertRaises(jsonschema.ValidationError):
            _ = Alias(name="")
        with self.assertRaises(jsonschema.ValidationError):
            _ = Alias(name="    ")

    def test_filter_name(self):
        for tgt, datasets in self.tgt2datasets.items():
            go_products = datasets[3]
            trained = Filter(pred=[it["Unit cost"] >= 10])
            transformed = trained.transform(go_products)
            self.assertEqual(get_table_name(transformed), "go_products", tgt)
            if tgt == "spark-with-index":
                self.assertEqual(get_index_name(transformed), "index", tgt)

    def test_map_name(self):
        for tgt, datasets in self.tgt2datasets.items():
            go_products = datasets[3]
            trained = Map(columns={"unit_cost": it["Unit cost"]})
            transformed = trained.transform(go_products)
            self.assertEqual(get_table_name(transformed), "go_products", tgt)
            if tgt == "spark-with-index":
                self.assertEqual(get_index_name(transformed), "index", tgt)

    def test_join_name(self):
        for tgt, datasets in self.tgt2datasets.items():
            trained = Join(
                pred=[it.go_1k["Retailer code"] == it.go_retailers["Retailer code"]],
                name="joined_tables",
            )
            transformed = trained.transform(datasets)
            self.assertEqual(get_table_name(transformed), "joined_tables", tgt)

    def test_groupby_name(self):
        for tgt, datasets in self.tgt2datasets.items():
            go_products = datasets[3]
            trained = GroupBy(by=[it["Product line"]])
            transformed = trained.transform(go_products)
            self.assertEqual(get_table_name(transformed), "go_products", tgt)

    def test_aggregate_name(self):
        for tgt, datasets in self.tgt2datasets.items():
            go_daily_sales = datasets[1]
            group_by = GroupBy(by=[it["Retailer code"]])
            aggregate = Aggregate(columns={"min_quantity": min(it.Quantity)})
            trained = group_by >> aggregate
            transformed = trained.transform(go_daily_sales)
            self.assertEqual(get_table_name(transformed), "go_daily_sales", tgt)


# Testing group_by operator for pandas and spark dataframes
class TestGroupBy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}

    def test_groupby(self):
        trainable = GroupBy(by=[it["Product line"]])
        for tgt, datasets in self.tgt2datasets.items():
            go_products = datasets[3]
            assert get_table_name(go_products) == "go_products"
            grouped_df = trainable.transform(go_products)
            if tgt == "pandas":
                self.assertEqual(grouped_df.ngroups, 5, tgt)
            aggregate = Aggregate(columns={"count": count(it["Product line"])})
            df = _ensure_pandas(aggregate.transform(grouped_df))
            self.assertEqual(df.shape, (5, 1), tgt)

    def test_groupby1(self):
        trainable = GroupBy(by=[it["Product line"], it.Product])
        for tgt, datasets in self.tgt2datasets.items():
            go_products = datasets[3]
            assert get_table_name(go_products) == "go_products"
            grouped_df = trainable.transform(go_products)
            if tgt == "pandas":
                self.assertEqual(grouped_df.ngroups, 144, tgt)
            aggregate = Aggregate(columns={"count": count(it["Product line"])})
            df = _ensure_pandas(aggregate.transform(grouped_df))
            self.assertEqual(df.shape, (144, 1), tgt)


# Testing Aggregate operator for both pandas and Spark
class TestAggregate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}

    def test_sales_not_grouped_single_col(self):
        pipeline = Scan(table=it.go_daily_sales) >> Aggregate(
            columns={
                "min_method_code": min(it["Order method code"]),
                "max_method_code": max(it["Order method code"]),
                "collect_set('Order method code')": collect_set(
                    it["Order method code"]
                ),
                "mode_method_code": mode(it["Order method code"]),
                "median_method_code": median(it["Order method code"]),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt.startswith("spark"):
                result = result.toPandas()
            self.assertEqual(result.shape, (1, 5), tgt)
            self.assertEqual(result.loc[0, "min_method_code"], 1, tgt)
            self.assertEqual(result.loc[0, "max_method_code"], 7, tgt)
            self.assertEqual(
                sorted(result.loc[0, "collect_set('Order method code')"]),
                [1, 2, 3, 4, 5, 6, 7],
                tgt,
            )
            self.assertEqual(result.loc[0, "mode_method_code"], 5, tgt)
            self.assertEqual(result.loc[0, "median_method_code"], 5, tgt)

    def test_sales_not_grouped_single_func(self):
        pipeline = Aggregate(
            columns={
                "max_method_code": max(it["Order method code"]),
                "max_method_type": max(it["Order method type"]),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets[2])
            if tgt.startswith("spark"):
                result = result.toPandas()
            self.assertEqual(result.shape, (1, 2), tgt)
            self.assertEqual(result.loc[0, "max_method_code"], 12, tgt)
            self.assertEqual(result.loc[0, "max_method_type"], "Web", tgt)

    def test_sales_multi_col_not_grouped(self):
        pipeline = Aggregate(
            columns={
                "min_method_code": min(it["Order method code"]),
                "max_method_code": max(it["Order method code"]),
                "max_method_type": max(it["Order method type"]),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets[2])
            if tgt.startswith("spark"):
                result = result.toPandas()
            self.assertEqual(result.shape, (1, 3), tgt)
            self.assertEqual(result.loc[0, "min_method_code"], 1, tgt)
            self.assertEqual(result.loc[0, "max_method_code"], 12, tgt)
            self.assertEqual(result.loc[0, "max_method_type"], "Web", tgt)

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
                    # Mode is not supported on GroupedData as of now.
                    # "mode_method_code": mode(it["Order method code"]),
                    "median_method_code": median(it["Order method code"]),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt.startswith("spark"):
                result = result.toPandas()
            self.assertEqual(result.shape, (289, 6))
            row = result[result.retailer_code == 1201]
            self.assertEqual(row.loc[row.index[0], "retailer_code"], 1201, tgt)
            self.assertEqual(row.loc[row.index[0], "min_method_code"], 2, tgt)
            self.assertEqual(row.loc[row.index[0], "max_method_code"], 6, tgt)
            self.assertEqual(row.loc[row.index[0], "min_quantity"], 1, tgt)
            self.assertEqual(
                sorted(row.loc[row.index[0], "method_codes"]), [2, 3, 4, 5, 6], tgt
            )
            # self.assertEqual(result.loc[row.index[0], "mode_method_code"], 5, tgt)
            self.assertEqual(result.loc[row.index[0], "median_method_code"], 5, tgt)
            self.assertEqual(result.index.name, "Retailer code", tgt)

    def test_sales_onekey_grouped_single_col(self):
        pipeline = (
            Scan(table=it.go_daily_sales)
            >> GroupBy(by=[it["Retailer code"]])
            >> Aggregate(
                columns={
                    "min_method_code": min(it["Order method code"]),
                    "max_method_code": max(it["Order method code"]),
                    "method_codes": collect_set(it["Order method code"]),
                    "median_method_code": median(it["Order method code"]),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt.startswith("spark"):
                result = result.toPandas()
            self.assertEqual(result.shape, (289, 4))
            self.assertEqual(result.loc[1201, "min_method_code"], 2, tgt)
            self.assertEqual(result.loc[1201, "max_method_code"], 6, tgt)
            self.assertEqual(
                sorted(result.loc[1201, "method_codes"]), [2, 3, 4, 5, 6], tgt
            )
            self.assertEqual(result.loc[1201, "median_method_code"], 5, tgt)
            self.assertEqual(result.index.name, "Retailer code", tgt)

    def test_sales_onekey_grouped_single_func(self):
        pipeline = (
            Scan(table=it.go_daily_sales)
            >> GroupBy(by=[it["Retailer code"]])
            >> Aggregate(
                columns={
                    "min_method_code": min(it["Order method code"]),
                    "min_quantity": min(it["Quantity"]),
                }
            )
        )
        for tgt, datasets in self.tgt2datasets.items():
            result = pipeline.transform(datasets)
            if tgt.startswith("spark"):
                result = result.toPandas()
            self.assertEqual(result.shape, (289, 2))
            self.assertEqual(result.loc[1201, "min_method_code"], 2, tgt)
            self.assertEqual(result.loc[1201, "min_quantity"], 1, tgt)
            self.assertEqual(result.index.name, "Retailer code", tgt)

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
            if tgt.startswith("spark"):
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
            result = _ensure_pandas(result)
            self.assertEqual(result.shape, (5000, 5))
            row = result[(result["product"] == 70240) & (result["retailer"] == 1205)]
            self.assertEqual(row.loc[row.index[0], "product"], 70240, tgt)
            self.assertEqual(row.loc[row.index[0], "retailer"], 1205, tgt)
            self.assertAlmostEqual(
                row.loc[row.index[0], "mean_quantity"], 48.39, 2, tgt
            )
            self.assertEqual(row.loc[row.index[0], "max_usp"], 122.70, tgt)
            self.assertEqual(row.loc[row.index[0], "count_quantity"], 41, tgt)
            self.assertEqual(
                result.index.names, ["Product number", "Retailer code"], tgt
            )

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
            if tgt.startswith("spark"):
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
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {
            tgt: {"go_sales": fetch_go_sales_dataset(tgt)} for tgt in targets
        }

        def add_df(name, df):
            cls.tgt2datasets["pandas"][name] = df
            cls.tgt2datasets["spark"][name] = pandas2spark(df)
            cls.tgt2datasets["spark-with-index"][name] = pandas2spark(
                df, with_index=True
            )

        table1 = {
            "train_id": [1, 2, 3, 4, 5],
            "col1": ["NY", "TX", "CA", "NY", "CA"],
            "col2": [0, 1, 1, 0, 1],
        }
        df1 = add_table_name(pd.DataFrame(data=table1), "main")
        add_df("df1", df1)

        table2 = {
            "TrainId": [1, 2, 3],
            "col3": ["USA", "USA", "UK"],
            "col4": [100, 100, 200],
        }
        df2 = add_table_name(pd.DataFrame(data=table2), "info")
        add_df("df2", df2)

        table3 = {
            "tid": [1, 2, 3],
            "col5": ["Warm", "Cold", "Warm"],
        }
        df3 = add_table_name(pd.DataFrame(data=table3), "t1")
        add_df("df3", df3)

        table4 = {
            "TrainId": [1, 2, 3, 4, 5],
            "col1": ["NY", "TX", "CA", "NY", "CA"],
            "col2": [0, 1, 1, 0, 1],
        }
        df4 = add_table_name(pd.DataFrame(data=table4), "main")
        add_df("df4", df4)

        table5 = {
            "TrainId": [1, 2, 3],
            "col3": ["NY", "NY", "CA"],
            "col4": [100, 100, 200],
        }
        df5 = add_table_name(pd.DataFrame(data=table5), "info")
        add_df("df5", df5)

        table6 = {
            "t_id": [2, 3],
            "col6": ["USA", "UK"],
        }
        df6 = add_table_name(pd.DataFrame(data=table6), "t2")
        add_df("df6", df6)

    # Multiple elements in predicate with different key column names
    def test_join_multiple_inner(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="inner",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df1, df2, df3 = datasets["df1"], datasets["df2"], datasets["df3"]
            transformed_df = trainable.transform([df1, df2, df3])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="train_id").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (3, 8), tgt)
            self.assertEqual(transformed_df["col5"][1], "Cold", tgt)

    # Multiple elements in predicate with identical key columns names
    def test_join_multiple_left(self):
        trainable = Join(
            pred=[it.main.TrainId == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="left",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df4, df2, df3 = datasets["df4"], datasets["df2"], datasets["df3"]
            transformed_df = trainable.transform([df4, df2, df3])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="TrainId").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (5, 7), tgt)
            self.assertEqual(transformed_df["col3"][2], "UK", tgt)

    # Invert one of the join conditions as compared to the test case: test_join_pandas_multiple_left
    def test_join_multiple_right(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.t1.tid == it.info.TrainId],
            join_type="right",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df1, df2, df3 = datasets["df1"], datasets["df2"], datasets["df3"]
            transformed_df = trainable.transform([df1, df2, df3])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="TrainId").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (3, 8), tgt)
            self.assertEqual(transformed_df["col3"][2], "UK", tgt)

    # Composite key join
    def test_join_composite(self):
        trainable = Join(
            pred=[
                it.t1.tid == it.info.TrainId,
                [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
            ],
            join_type="left",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df1, df5, df3, df6 = (
                datasets["df1"],
                datasets["df5"],
                datasets["df3"],
                datasets["df6"],
            )
            transformed_df = trainable.transform([df1, df5, df3, df6])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="train_id").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (5, 8), tgt)
            self.assertEqual(transformed_df["col3"][2], "CA", tgt)

    # Invert one of the join conditions as compared to the test case: test_join_pandas_composite
    def test_join_composite1(self):
        trainable = Join(
            pred=[
                [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                it.t1.tid == it.info.TrainId,
                it.t1.tid == it.t2.t_id,
            ],
            join_type="inner",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df1, df5, df3, df6 = (
                datasets["df1"],
                datasets["df5"],
                datasets["df3"],
                datasets["df6"],
            )
            transformed_df = trainable.transform([df1, df5, df3, df6])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="train_id").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (1, 10), tgt)
            self.assertEqual(transformed_df["col4"][0], 200, tgt)

    # Composite key join having conditions involving more than 2 tables
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_composite_error(self):
        with self.assertRaisesRegex(
            ValueError, "info.*main.*inFo.* more than two tables"
        ):
            _ = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.inFo.col3],
                    it.t1.tid == it.t2.t_id,
                ],
                join_type="inner",
            )

    # Single joining conditions are not chained
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_single_error1(self):
        with self.assertRaisesRegex(ValueError, "t3.*t2.* were used"):
            _ = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                    it.t3.tid == it.t2.t_id,
                ],
                join_type="inner",
            )

    def test_join_composite_nochain_error(self):
        with self.assertRaisesRegex(ValueError, "t3.*t2.* were used"):
            _ = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.main.col1 == it.info.col3],
                    [it.t3.tid == it.t2.t_id, it.t3.TrainId == it.t2.TrainId],
                ],
                join_type="inner",
            )
            # _ = trainable.transform([self.df1, self.df5, self.df3, self.df6])

    # Composite key join having conditions involving more than 2 tables
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_composite_error2(self):
        with self.assertRaisesRegex(
            ValueError, "main.*info.*Main.*inFo.*more than two"
        ):
            _ = Join(
                pred=[
                    it.t1.tid == it.info.TrainId,
                    [it.main.train_id == it.info.TrainId, it.Main.col1 == it.inFo.col3],
                    it.t1.tid == it.t2.t_id,
                ],
                join_type="inner",
            )

    # A table to be joined not present in input X
    # This test case execution should throw a ValueError which is handled in the test case itself
    def test_join_composite_error3(self):
        for tgt, datasets in self.tgt2datasets.items():
            df5, df3 = datasets["df5"], datasets["df3"]
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
                _ = trainable.transform([df5, df3])

    # TestCase 1: Go_Sales dataset with different forms of predicate (join conditions)
    def test_join_go_sales1(self):
        for tgt, datasets in self.tgt2datasets.items():
            go_sales = datasets["go_sales"]
            trainable = Join(
                pred=[
                    it.go_daily_sales["Retailer code"]
                    == it["go_retailers"]["Retailer code"]
                ],
                join_type="inner",
            )
            transformed_df = trainable.transform(go_sales)
            order = ["Retailer code", "Product number", "Date"]
            if tgt == "pandas":
                transformed_df = transformed_df.sort_values(by=order).reset_index(
                    drop=True
                )
                self.assertEqual(transformed_df.shape, (149257, 10), tgt)
                self.assertEqual(transformed_df["Country"][4], "France", tgt)
            elif tgt.startswith("spark"):
                self.assertEqual(len(get_columns(transformed_df)), 10, tgt)
                self.assertEqual(transformed_df.count(), 149257, tgt)
                # transformed_df = transformed_df.orderBy(order).collect()
                # self.assertEqual(transformed_df[4]["Country"], "France", tgt)
            else:
                assert False

    # TestCase 2: Go_Sales dataset throws error because of duplicate non-key columns
    def test_join_go_sales2(self):
        for tgt, datasets in self.tgt2datasets.items():
            go_sales = datasets["go_sales"]
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
            with self.assertRaises(ValueError):
                _ = trainable.transform(go_sales)

    def test_join_index(self):
        trainable = Join(
            pred=[it.info.idx == it.main.idx, it.info.idx == it.t1.idx],
            join_type="inner",
        )
        df1 = _set_index_name(self.tgt2datasets["pandas"]["df1"], "idx")
        df2 = _set_index_name(self.tgt2datasets["pandas"]["df2"], "idx")
        df3 = _set_index_name(self.tgt2datasets["pandas"]["df3"], "idx")
        for tgt in ["spark-with-index"]:
            if tgt == "spark-with-index":
                df1 = pandas2spark(df1, with_index=True)
                df2 = pandas2spark(df2, with_index=True)
                df3 = pandas2spark(df3, with_index=True)
            transformed_df = trainable.transform([df1, df2, df3])
            transformed_df = _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.index.name, "idx", tgt)
            transformed_df = transformed_df.sort_values(by="TrainId").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (3, 8), tgt)
            self.assertEqual(transformed_df["col5"][1], "Cold", tgt)

    def test_join_one_index_right(self):
        trainable = Join(
            pred=[it.info.TrainId == it.main.train_id, it.info.TrainId == it.t1.tid],
            join_type="inner",
        )
        df1 = _set_index(self.tgt2datasets["pandas"]["df1"], "train_id")
        df2 = self.tgt2datasets["pandas"]["df2"]
        df3 = self.tgt2datasets["pandas"]["df3"]
        for tgt in ["spark-with-index"]:
            if tgt == "spark-with-index":
                df1 = pandas2spark(df1, with_index=True)
                df2 = pandas2spark(df2, with_index=True)
                df3 = pandas2spark(df3)
            transformed_df = trainable.transform([df1, df2, df3])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="TrainId").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (3, 7))
            self.assertEqual(transformed_df["col5"][1], "Cold")

    def test_join_one_index_left(self):
        trainable = Join(
            pred=[it.main.train_id == it.info.TrainId, it.info.TrainId == it.t1.tid],
            join_type="inner",
        )
        df1 = _set_index(self.tgt2datasets["pandas"]["df1"], "train_id")
        df2 = self.tgt2datasets["pandas"]["df2"]
        df3 = self.tgt2datasets["pandas"]["df3"]
        for tgt in ["spark-with-index"]:
            if tgt == "spark-with-index":
                df1 = pandas2spark(df1, with_index=True)
                df2 = pandas2spark(df2, with_index=True)
                df3 = pandas2spark(df3)
            transformed_df = trainable.transform([df1, df2, df3])
            transformed_df = _ensure_pandas(transformed_df)
            transformed_df = transformed_df.sort_values(by="TrainId").reset_index(
                drop=True
            )
            self.assertEqual(transformed_df.shape, (3, 7))
            self.assertEqual(transformed_df["col5"][1], "Cold")

    def test_join_index_multiple_names(self):
        trainable = Join(
            pred=[it.info.TrainId == it.main.train_id, it.info.TrainId == it.t1.tid],
            join_type="inner",
        )
        df1 = _set_index(self.tgt2datasets["pandas"]["df1"], "train_id")
        df2 = _set_index(self.tgt2datasets["pandas"]["df2"], "TrainId")
        df3 = _set_index(self.tgt2datasets["pandas"]["df3"], "tid")
        for tgt in ["spark-with-index"]:
            if tgt == "spark-with-index":
                df1 = pandas2spark(df1, with_index=True)
                df2 = pandas2spark(df2, with_index=True)
                df3 = pandas2spark(df3, with_index=True)
        transformed_df = trainable.transform([df1, df2, df3])
        transformed_df = _ensure_pandas(transformed_df)
        transformed_df = transformed_df.sort_values(by="TrainId").reset_index(drop=True)
        self.assertEqual(transformed_df.shape, (3, 6))
        self.assertEqual(transformed_df["col5"][1], "Cold")


class TestMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {
            tgt: {"go_sales": fetch_go_sales_dataset(tgt)} for tgt in targets
        }

        def add_df(name, df):
            cls.tgt2datasets["pandas"][name] = df
            cls.tgt2datasets["spark"][name] = pandas2spark(df)
            cls.tgt2datasets["spark-with-index"][name] = pandas2spark(
                df, with_index=True
            )

        df = pd.DataFrame(
            {
                "gender": ["m", "f", "m", "m", "f"],
                "state": ["NY", "NY", "CA", "NY", "CA"],
                "status": [0, 1, 1, 0, 1],
            }
        )
        add_df("df", df)
        df_date = pd.DataFrame(
            {"date_column": ["2016-05-28", "2016-06-27", "2016-07-26"]}
        )
        add_df("df_date", df_date)
        df_date_alt = pd.DataFrame(
            {"date_column": ["28/05/2016", "27/06/2016", "26/07/2016"]}
        )
        add_df("df_date_alt", df_date_alt)
        df_date_time = pd.DataFrame(
            {
                "date_column": [
                    "2016-01-01 15:16:45",
                    "2016-06-28 12:18:51",
                    "2016-07-28 01:01:01",
                ]
            }
        )
        add_df("df_date_time", df_date_time)
        df_num = pd.DataFrame(
            {
                "height": [3, 4, 6, 3, 5],
                "weight": [30, 50, 170, 40, 130],
                "status": [0, 1, 1, 0, 1],
            }
        )
        add_df("df_num", df_num)
        df_month = pd.DataFrame(
            {
                "month": ["jan", "feb", "mar", "may", "aug"],
            }
        )
        add_df("df_month", df_month)

    def test_init(self):
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        _ = Map(columns=[replace(it.gender, gender_map), replace(it.state, state_map)])

    # The rename column functionality implemented as part of identity function for Map operator
    # does not support explicit identity calls for now.
    def test_transform_identity_map(self):
        trainable = Map(
            columns={
                "new_gender": it.gender,
                "new_status": it["status"],
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(df["gender"][0], transformed_df["new_gender"][0], tgt)
            self.assertEqual(df["status"][3], transformed_df["new_status"][3], tgt)
            self.assertEqual(len(transformed_df.columns), 2, tgt)

    def test_transform_identity_map_implicit_name(self):
        trainable = Map(columns=[identity(it.gender), identity(it["status"])])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(df["gender"][0], transformed_df["gender"][0], tgt)
            self.assertEqual(df["status"][3], transformed_df["status"][3], tgt)
            self.assertEqual(len(transformed_df.columns), 2, tgt)

    def test_transform_identity_map_passthrough(self):
        trainable = Map(
            columns={
                "new_gender": it.gender,
                "new_status": it["status"],
            },
            remainder="passthrough",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(df["gender"][0], transformed_df["new_gender"][0])
            self.assertEqual(df["status"][3], transformed_df["new_status"][3])
            self.assertEqual(df["state"][3], transformed_df["state"][3])
            self.assertEqual(len(transformed_df.columns), 3)

    def test_transform_identity_map_error(self):
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
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
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        trainable = Map(
            columns=[replace(it.gender, gender_map), replace(it.state, state_map)],
            remainder="passthrough",
        )
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 3))
            self.assertEqual(transformed_df["gender"][0], "Male")
            self.assertEqual(transformed_df["state"][0], "New York")
            self.assertEqual(transformed_df["status"][0], 0)

    def test_transform_replace_list(self):
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        trainable = Map(
            columns=[replace(it.gender, gender_map), replace(it.state, state_map)]
        )
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 2))
            self.assertEqual(transformed_df["gender"][0], "Male")
            self.assertEqual(transformed_df["state"][0], "New York")

    def test_transform_replace_map(self):
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        trainable = Map(
            columns={
                "new_gender": replace(it.gender, gender_map),
                "new_state": replace(it.state, state_map),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 2))
            self.assertEqual(transformed_df["new_gender"][0], "Male")
            self.assertEqual(transformed_df["new_state"][0], "New York")

    def test_transform_dom_list(self):
        trainable = Map(columns=[day_of_month(it.date_column)])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 28)
            self.assertEqual(transformed_df["date_column"][1], 27)
            self.assertEqual(transformed_df["date_column"][2], 26)

    def test_transform_dom_fmt_list(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(columns=[day_of_month(it.date_column, "%Y-%m-%d")])
            elif tgt.startswith("spark"):
                trainable = Map(columns=[day_of_month(it.date_column, "y-M-d")])
            else:
                assert False
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 28)
            self.assertEqual(transformed_df["date_column"][1], 27)
            self.assertEqual(transformed_df["date_column"][2], 26)

    def test_transform_dom_fmt_map(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={"dom": day_of_month(it.date_column, "%Y-%m-%d")}
                )
            elif tgt.startswith("spark"):
                trainable = Map(columns={"dom": day_of_month(it.date_column, "y-M-d")})
            else:
                assert False
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (3, 1))
            self.assertEqual(transformed_df["dom"][0], 28)
            self.assertEqual(transformed_df["dom"][1], 27)
            self.assertEqual(transformed_df["dom"][2], 26)

    def test_transform_dow_list(self):
        trainable = Map(columns=[day_of_week(it.date_column)])
        for tgt, datasets in self.tgt2datasets.items():
            # Spark and Pandas have a different semantics for `day_of_week`
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            if tgt == "pandas":
                self.assertEqual(transformed_df["date_column"][0], 5)
                self.assertEqual(transformed_df["date_column"][1], 0)
                self.assertEqual(transformed_df["date_column"][2], 1)
            elif tgt.startswith("spark"):
                df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
                self.assertEqual(transformed_df["date_column"][0], 7)
                self.assertEqual(transformed_df["date_column"][1], 2)
                self.assertEqual(transformed_df["date_column"][2], 3)
            else:
                assert False

    def test_transform_dow_fmt_list(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(columns=[day_of_week(it.date_column, "%Y-%m-%d")])
                df = datasets["df_date"]
                trained = trainable.fit(df)
                transformed_df = trained.transform(df)
                self.assertEqual(transformed_df.shape, (3, 1))
                self.assertEqual(transformed_df["date_column"][0], 5)
                self.assertEqual(transformed_df["date_column"][1], 0)
                self.assertEqual(transformed_df["date_column"][2], 1)
            elif tgt.startswith("spark"):
                trainable = Map(columns=[day_of_week(it.date_column, "y-M-d")])
                df = datasets["df_date"]
                trained = trainable.fit(df)
                transformed_df = trained.transform(df)
                df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
                self.assertEqual(transformed_df.shape, (3, 1))
                self.assertEqual(transformed_df["date_column"][0], 7)
                self.assertEqual(transformed_df["date_column"][1], 2)
                self.assertEqual(transformed_df["date_column"][2], 3)
            else:
                assert False

    def test_transform_dow_fmt_map(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={"dow": day_of_week(it.date_column, "%Y-%m-%d")}
                )
                df = datasets["df_date"]
                trained = trainable.fit(df)
                transformed_df = trained.transform(df)
                df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
                self.assertEqual(transformed_df.shape, (3, 1))
                self.assertEqual(transformed_df["dow"][0], 5)
                self.assertEqual(transformed_df["dow"][1], 0)
                self.assertEqual(transformed_df["dow"][2], 1)
            elif tgt.startswith("spark"):
                trainable = Map(columns={"dow": day_of_week(it.date_column, "y-M-d")})
                df = datasets["df_date"]
                trained = trainable.fit(df)
                transformed_df = trained.transform(df)
                df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
                self.assertEqual(transformed_df.shape, (3, 1))
                self.assertEqual(transformed_df["dow"][0], 7)
                self.assertEqual(transformed_df["dow"][1], 2)
                self.assertEqual(transformed_df["dow"][2], 3)
            else:
                assert False

    def test_transform_doy_list(self):
        trainable = Map(columns=[day_of_year(it.date_column)])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 149)
            self.assertEqual(transformed_df["date_column"][1], 179)
            self.assertEqual(transformed_df["date_column"][2], 208)

    def test_transform_doy_fmt_list(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(columns=[day_of_year(it.date_column, "%Y-%m-%d")])
            elif tgt.startswith("spark"):
                trainable = Map(columns=[day_of_year(it.date_column, "y-M-d")])
            else:
                assert False
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 149)
            self.assertEqual(transformed_df["date_column"][1], 179)
            self.assertEqual(transformed_df["date_column"][2], 208)

    def test_transform_doy_fmt_map(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={"doy": day_of_year(it.date_column, "%Y-%m-%d")}
                )
            elif tgt.startswith("spark"):
                trainable = Map(columns={"doy": day_of_year(it.date_column, "y-M-d")})
            else:
                assert False
            df = datasets["df_date"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (3, 1))
            self.assertEqual(transformed_df["doy"][0], 149)
            self.assertEqual(transformed_df["doy"][1], 179)
            self.assertEqual(transformed_df["doy"][2], 208)

    def test_transform_hour_list(self):
        trainable = Map(columns=[hour(it.date_column)])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 15)
            self.assertEqual(transformed_df["date_column"][1], 12)
            self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_hour_fmt_list(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(columns=[hour(it.date_column, "%Y-%m-%d %H:%M:%S")])
            elif tgt.startswith("spark"):
                trainable = Map(columns=[hour(it.date_column, "y-M-d HH:mm:ss")])
            else:
                assert False
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 15)
            self.assertEqual(transformed_df["date_column"][1], 12)
            self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_hour_fmt_map(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={"hour": hour(it.date_column, "%Y-%m-%d %H:%M:%S")}
                )
            elif tgt.startswith("spark"):
                trainable = Map(
                    columns={"hour": hour(it.date_column, "y-M-d HH:mm:ss")}
                )
            else:
                assert False
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (3, 1))
            self.assertEqual(transformed_df["hour"][0], 15)
            self.assertEqual(transformed_df["hour"][1], 12)
            self.assertEqual(transformed_df["hour"][2], 1)

    def test_transform_minute_list(self):
        trainable = Map(columns=[minute(it.date_column)])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 16)
            self.assertEqual(transformed_df["date_column"][1], 18)
            self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_minute_fmt_list(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(columns=[minute(it.date_column, "%Y-%m-%d %H:%M:%S")])
            elif tgt.startswith("spark"):
                trainable = Map(columns=[minute(it.date_column, "y-M-d HH:mm:ss")])
            else:
                assert False
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 16)
            self.assertEqual(transformed_df["date_column"][1], 18)
            self.assertEqual(transformed_df["date_column"][2], 1)

    def test_transform_minute_fmt_map(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={"minute": minute(it.date_column, "%Y-%m-%d %H:%M:%S")}
                )
            elif tgt.startswith("spark"):
                trainable = Map(
                    columns={"minute": minute(it.date_column, "y-M-d HH:mm:ss")}
                )
            else:
                assert False
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (3, 1))
            self.assertEqual(transformed_df["minute"][0], 16)
            self.assertEqual(transformed_df["minute"][1], 18)
            self.assertEqual(transformed_df["minute"][2], 1)

    def test_transform_month_list(self):
        trainable = Map(columns=[month(it.date_column)])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 1)
            self.assertEqual(transformed_df["date_column"][1], 6)
            self.assertEqual(transformed_df["date_column"][2], 7)

    def test_transform_month_fmt_list(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(columns=[month(it.date_column, "%Y-%m-%d %H:%M:%S")])
            elif tgt.startswith("spark"):
                trainable = Map(columns=[month(it.date_column, "y-M-d HH:mm:ss")])
            else:
                assert False
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date_column"][0], 1)
            self.assertEqual(transformed_df["date_column"][1], 6)
            self.assertEqual(transformed_df["date_column"][2], 7)

    def test_transform_month_fmt_map(self):
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={"month": month(it.date_column, "%Y-%m-%d %H:%M:%S")}
                )
            elif tgt.startswith("spark"):
                trainable = Map(
                    columns={"month": month(it.date_column, "y-M-d HH:mm:ss")}
                )
            else:
                assert False
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (3, 1))
            self.assertEqual(transformed_df["month"][0], 1)
            self.assertEqual(transformed_df["month"][1], 6)
            self.assertEqual(transformed_df["month"][2], 7)

    def test_not_expression(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = Map(columns=[123, "hello"])

    def test_pandas_with_hyperopt(self):
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

    def test_string_indexer_map(self):
        trainable = Map(columns={"c": string_indexer(it.date_column)})
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_date_time"]
            trained = trainable.fit(df)
            with self.assertRaises(ValueError):
                _ = trained.transform(df)

    def test_pands_with_hyperopt2(self):
        from lale.expressions import count, it, max, mean, min, sum, variance

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
            operator=lale.operators.make_pipeline_graph(
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

        opt = Hyperopt(estimator=pipeline, max_evals=2)
        opt.fit(X, y)

    def test_transform_ratio_map(self):
        trainable = Map(columns={"ratio_h_w": it.height / it.weight})
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 1))
            self.assertEqual(transformed_df["ratio_h_w"][0], 0.1)

    def test_transform_ratio_map_subscript(self):
        trainable = Map(columns={"ratio_h_w": it["height"] / it.weight})
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 1))
            self.assertEqual(transformed_df["ratio_h_w"][0], 0.1)

    def test_transform_ratio_map_list(self):
        trainable = Map(columns=[it.height / it.weight])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            with self.assertRaises(ValueError):
                _ = trained.transform(df)

    def test_transform_subtract_map(self):
        trainable = Map(columns={"subtract_h_w": it.height - it.weight})
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 1))
            self.assertEqual(transformed_df["subtract_h_w"][0], -27)

    def test_transform_subtract_map_subscript(self):
        trainable = Map(columns={"subtract_h_w": it["height"] - it.weight})
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 1))
            self.assertEqual(transformed_df["subtract_h_w"][0], -27)

    def test_transform_subtract_map_list(self):
        trainable = Map(columns=[it.height - it.weight])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            with self.assertRaises(ValueError):
                _ = trained.transform(df)

    def test_transform_binops(self):
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
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
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
            if tgt == "pandas":
                self.assertEqual(
                    transformed_df["pow_h_w"][1], df["height"][1] ** df["weight"][1]
                )
            elif tgt.startswith("spark"):
                # Spark and Pandas have a different semantics for large numbers
                self.assertEqual(transformed_df["pow_h_w"][1], 4**50)
            else:
                assert False
            self.assertEqual(transformed_df["pow_h_2"][1], df["height"][1] ** 2)

    def test_transform_arithmetic_expression(self):
        trainable = Map(columns={"expr": (it["height"] + it.weight * 10) / 2})
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df.shape, (5, 1))
            self.assertEqual(
                transformed_df["expr"][2], (df["height"][2] + df["weight"][2] * 10) / 2
            )

    def test_transform_nested_expressions(self):
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
        for tgt, datasets in self.tgt2datasets.items():
            if tgt == "pandas":
                trainable = Map(
                    columns={
                        "date": replace(it.month, month_map),
                        "month_id": month(replace(it.month, month_map), "%Y-%m-%d"),
                        "next_month_id": identity(
                            month(replace(it.month, month_map), "%Y-%m-%d") % 12 + 1
                        ),
                    }
                )
            elif tgt.startswith("spark"):
                trainable = Map(
                    columns={
                        "date": replace(it.month, month_map),
                        "month_id": month(replace(it.month, month_map), "y-M-d"),
                        "next_month_id": identity(
                            month(replace(it.month, month_map), "y-M-d") % 12 + 1
                        ),
                    }
                )
            else:
                assert False

            df = datasets["df_month"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertEqual(transformed_df["date"][0], "2021-01-01")
            self.assertEqual(transformed_df["date"][1], "2021-02-01")
            self.assertEqual(transformed_df["month_id"][2], 3)
            self.assertEqual(transformed_df["month_id"][3], 5)
            self.assertEqual(transformed_df["next_month_id"][0], 2)
            self.assertEqual(transformed_df["next_month_id"][3], 6)
            self.assertEqual(transformed_df["next_month_id"][4], 9)

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
            result = pipeline.transform(datasets["go_sales"])
            result = _ensure_pandas(result)
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
            result = pipeline.transform(datasets["go_sales"])
            result = _ensure_pandas(result)
            self.assertEqual(result.shape, (274, 2))
            self.assertEqual(result.loc[0, "prod"], 1110, tgt)
            self.assertEqual(result.loc[0, "line"], "C", tgt)
            self.assertEqual(result.loc[117, "prod"], 101110, tgt)
            self.assertEqual(result.loc[117, "line"], "U", tgt)
            self.assertEqual(result.loc[273, "prod"], 154150, tgt)
            self.assertEqual(result.loc[273, "line"], "P", tgt)

    def test_dynamic_rename(self):
        def expr(X):
            return {("new_" + c): it[c] for c in X.columns}

        pipeline = Scan(table=it.go_products) >> Map(columns=expr)
        for tgt, datasets in self.tgt2datasets.items():
            datasets = datasets["go_sales"]
            result = pipeline.fit(datasets).transform(datasets)
            result = _ensure_pandas(result)
            for c in result.columns:
                self.assertRegex(c, "new_.*")

    def test_dynamic_rename_lambda(self):
        pipeline = Scan(table=it.go_products) >> Map(
            columns=lambda X: {("new_" + c): it[c] for c in X.columns}
        )
        for tgt, datasets in self.tgt2datasets.items():
            datasets = datasets["go_sales"]
            result = pipeline.fit(datasets).transform(datasets)
            result = _ensure_pandas(result)
            for c in result.columns:
                self.assertRegex(c, "new_.*")

    def _get_col_schemas(self, cols, X):
        from lale.datasets import data_schemas

        props = {}
        s = data_schemas.to_schema(X)
        if s is not None:
            inner = s.get("items", {})
            if inner is not None and isinstance(inner, dict):
                col_pairs = inner.get("items", [])
                if col_pairs is not None and isinstance(col_pairs, list):
                    for cp in col_pairs:
                        d = cp.get("description", None)
                        if d is not None and isinstance(d, str):
                            props[d] = cp
        for k in cols:
            if k not in props:
                props[k] = None
        return props

    def test_dynamic_schema_num(self):
        from lale import type_checking

        def expr(X):
            ret = {}
            schemas = self._get_col_schemas(X.columns, X)
            for c, s in schemas.items():

                if s is None:
                    ret["unknown_" + c] = it[c]
                elif type_checking.is_subschema(s, {"type": "number"}):
                    ret["num_" + c] = it[c]
                    ret["shifted_" + c] = it[c] + 5
                else:
                    ret["other_" + c] = it[c]
            return ret

        pipeline = Scan(table=it.go_products) >> Map(columns=expr)
        for tgt, datasets in self.tgt2datasets.items():
            datasets = datasets["go_sales"]
            result = pipeline.fit(datasets).transform(datasets)
            result = _ensure_pandas(result)
            self.assertIn("num_Product number", result.columns)
            self.assertIn("shifted_Product number", result.columns)
            self.assertIn("other_Product line", result.columns)
            self.assertEqual(
                result["num_Product number"][0] + 5, result["shifted_Product number"][0]
            )

    def test_dynamic_categorical(self):
        from lale.lib.rasl import categorical

        def expr(X):
            ret = {}
            cats = categorical()(X)
            for c in X.columns:
                if c in cats:
                    ret["cat_" + c] = it[c]
                else:
                    ret["other_" + c] = it[c]
            return ret

        pipeline = Scan(table=it.go_products) >> Map(columns=expr)
        for tgt, datasets in self.tgt2datasets.items():
            datasets = datasets["go_sales"]
            result = pipeline.fit(datasets).transform(datasets)
            result = _ensure_pandas(result)
            self.assertIn("cat_Product line", result.columns)

    def test_dynamic_lambda_categorical_drop(self):
        from lale.lib.rasl import categorical

        pipeline = Scan(table=it.go_products) >> Map(
            columns=lambda X: {c: it[c] for c in categorical()(_ensure_pandas(X))}
        )
        for tgt, datasets in self.tgt2datasets.items():
            datasets = datasets["go_sales"]
            result = pipeline.fit(datasets).transform(datasets)
            result = _ensure_pandas(result)
            self.assertEqual(len(result.columns), 1)
            self.assertIn("Product line", result.columns)

    def test_static_trained(self):
        op = Map(columns=[it.col])
        self.assertIsInstance(op, lale.operators.TrainedOperator)

    def test_dynamic_trainable(self):
        op = Map(columns=lambda X: [it.col])
        self.assertIsInstance(op, lale.operators.TrainableOperator)
        self.assertNotIsInstance(op, lale.operators.TrainedOperator)

        pipeline = Scan(table=it.go_products) >> op
        pd = self.tgt2datasets["pandas"]["go_sales"]
        trained = pipeline.fit(pd)
        trained_map = trained.steps_list()[1]
        self.assertIsInstance(trained_map, Map)  # type: ignore
        self.assertIsInstance(trained_map, lale.operators.TrainedOperator)

    def test_project(self):
        from lale.lib.lale import Project

        pipeline = Scan(table=it.go_products) >> Project(columns={"type": "number"})
        for tgt, datasets in self.tgt2datasets.items():
            datasets = datasets["go_sales"]
            result = pipeline.fit(datasets).transform(datasets)
            result = _ensure_pandas(result)
            self.assertIn("Product number", result.columns)
            self.assertNotIn("Product line", result.columns)

    def assertSeriesEqual(self, first, second, msg=None):
        self.assertIsInstance(first, pd.Series, msg)
        self.assertIsInstance(second, pd.Series, msg)
        self.assertEqual(first.shape, second.shape, msg)
        self.assertEqual(list(first), list(second), msg)

    def test_transform_compare_ops(self):
        trained = Map(
            columns={
                "height<=5": it.height <= 5,
                "int(height<=5)": astype("int", it.height <= 5),
                "4==height": 4 == it.height,
                "height*10==weight": it.height * 10 == it.weight,
                "height>3&<=5": (it.height > 3) & (it.height <= 5),
                "height<=3|>5": (it.height <= 3) | (it.height > 5),
            }
        )
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df_num"]
            transformed_df = trained.transform(df)
            df, transformed_df = _ensure_pandas(df), _ensure_pandas(transformed_df)
            self.assertSeriesEqual(transformed_df["height<=5"], df["height"] <= 5, tgt)
            self.assertSeriesEqual(
                transformed_df["int(height<=5)"], (df["height"] <= 5).astype(int), tgt
            )
            self.assertSeriesEqual(transformed_df["4==height"], 4 == df["height"], tgt)
            self.assertSeriesEqual(
                transformed_df["height*10==weight"], df["height"] * 10 == df.weight, tgt
            )
            self.assertSeriesEqual(
                transformed_df["height>3&<=5"], (df["height"] > 3) & (df["height"] <= 5)
            )
            self.assertSeriesEqual(
                transformed_df["height<=3|>5"], (df["height"] <= 3) | (df["height"] > 5)
            )


class TestRelationalOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {tgt: {} for tgt in targets}

        def add_df(name, df):
            cls.tgt2datasets["pandas"][name] = df
            cls.tgt2datasets["spark"][name] = pandas2spark(df)
            cls.tgt2datasets["spark-with-index"][name] = pandas2spark(
                df, with_index=True
            )

        X, y = load_iris(as_frame=True, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        add_df("X_train", X_train)
        add_df("X_test", X_test)
        add_df("y_train", y_train)
        add_df("y_test", y_test)

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
        for tgt, datasets in self.tgt2datasets.items():
            X_train, X_test, y_train = (
                datasets["X_train"],
                datasets["X_test"],
                datasets["y_train"],
            )
            trained_relational = relational.fit(X_train, y_train)
            _ = trained_relational.transform(X_test)

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
        for tgt, datasets in self.tgt2datasets.items():
            X_train, y_train = datasets["X_train"], datasets["y_train"]
            with self.assertRaises(ValueError):
                _ = relational.fit([X_train], y_train)

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
        for tgt, datasets in self.tgt2datasets.items():
            X_train, X_test, y_train = (
                datasets["X_train"],
                datasets["X_test"],
                datasets["y_train"],
            )
            trained_relational = relational.fit(X_train, y_train)
            with self.assertRaises(ValueError):
                _ = trained_relational.transform([X_test])

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
        for tgt, datasets in self.tgt2datasets.items():
            X_train, X_test, y_train = (
                datasets["X_train"],
                datasets["X_test"],
                datasets["y_train"],
            )
            if tgt == "pandas":
                trained_pipeline = pipeline.fit(X_train, y_train)
                _ = trained_pipeline.predict(X_test)
            elif tgt.startswith("spark"):
                # LogisticRegression is not implemented on Spark
                pass
            else:
                assert False


class TestOrderBy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {tgt: {} for tgt in targets}

        def add_df(name, df):
            cls.tgt2datasets["pandas"][name] = df
            cls.tgt2datasets["spark"][name] = pandas2spark(df)
            cls.tgt2datasets["spark-with-index"][name] = pandas2spark(
                df, with_index=True
            )

        df = pd.DataFrame(
            {
                "gender": ["m", "f", "m", "m", "f"],
                "state": ["NY", "NY", "CA", "NY", "CA"],
                "status": [0, 1, 1, 0, 1],
            }
        )
        add_df("df", df)

    def test_order_attr1(self):
        trainable = OrderBy(by=it.status)
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            if tgt == "spark-with-index":
                self.assertEqual(get_index_name(transformed_df), get_index_name(df))
            transformed_df = _ensure_pandas(transformed_df)
            self.assertTrue((transformed_df["status"]).is_monotonic_increasing)

    def test_order_attr1_asc(self):
        trainable = OrderBy(by=asc(it.status))
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            if tgt == "spark-with-index":
                self.assertEqual(get_index_name(transformed_df), get_index_name(df))
            transformed_df = _ensure_pandas(transformed_df)
            self.assertTrue((transformed_df["status"]).is_monotonic_increasing)

    def test_order_attr1_desc(self):
        trainable = OrderBy(by=desc(it.status))
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            transformed_df = _ensure_pandas(transformed_df)
            self.assertTrue((transformed_df["status"]).is_monotonic_decreasing)

    def test_order_str1_desc(self):
        trainable = OrderBy(by=desc("gender"))
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            transformed_df = _ensure_pandas(transformed_df)
            self.assertTrue((transformed_df["gender"]).is_monotonic_decreasing)

    def test_order_multiple(self):
        trainable = OrderBy(by=[it.gender, desc(it.status)])
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            transformed_df = _ensure_pandas(transformed_df)
            expected_result = pd.DataFrame(
                data={
                    "gender": ["f", "f", "m", "m", "m"],
                    "state": ["NY", "CA", "CA", "NY", "NY"],
                    "status": [1, 1, 1, 0, 0],
                }
            )
            if tgt == "pandas" or tgt == "spark-with-index":
                self.assertEqual(list(transformed_df.index), [1, 4, 2, 0, 3])
            self.assertTrue((transformed_df["gender"]).is_monotonic_increasing)
            self.assertTrue(
                transformed_df.reset_index(drop=True).equals(expected_result)
            )

    def test_str1(self):
        trainable = OrderBy(by="gender")
        for tgt, datasets in self.tgt2datasets.items():
            df = datasets["df"]
            trained = trainable.fit(df)
            transformed_df = trained.transform(df)
            transformed_df = _ensure_pandas(transformed_df)
            self.assertTrue((transformed_df["gender"]).is_monotonic_increasing)


class TestSplitXy(unittest.TestCase):
    @classmethod
    def setUp(cls):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            pd.DataFrame(X), pd.DataFrame(y)
        )
        combined_df = pd.concat([X_train, y_train], axis=1)
        combined_df.columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]
        spark_df = pandas2spark(combined_df)
        cls.tgt2datasets = {
            "pandas": combined_df,
            "spark": spark_df,
            "spark-with-index": SparkDataFrameWithIndex(spark_df),
        }

    def test_split_transform(self):
        for _, df in self.tgt2datasets.items():
            trainable = SplitXy(label_name="class") >> Convert(astype="pandas") >> PCA()
            trained = trainable.fit(df)
            _ = trained.transform(df)

    def test_split_predict(self):
        for _, df in self.tgt2datasets.items():
            trainable = (
                SplitXy(label_name="class")
                >> Convert(astype="pandas")
                >> PCA()
                >> LogisticRegression(random_state=42)
            )
            trained = trainable.fit(df)
            _ = trained.predict(df)


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


class TestConvert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark", "spark-with-index"]
        cls.tgt2datasets = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}

    def _check(self, src, dst, tgt):
        self.assertEqual(get_table_name(src), get_table_name(dst), tgt)
        self.assertEqual(list(get_columns(src)), list(get_columns(dst)), tgt)
        pd_src = _ensure_pandas(src)
        pd_dst = _ensure_pandas(dst)
        self.assertEqual(pd_src.shape, pd_dst.shape, tgt)

    def test_to_pandas(self):
        for tgt, datasets in self.tgt2datasets.items():
            transformer = Convert(astype="pandas")
            go_products = datasets[3]
            self.assertEqual(get_table_name(go_products), "go_products", tgt)
            transformed_df = transformer.transform(go_products)
            self.assertTrue(_is_pandas_df(transformed_df), tgt)
            self._check(go_products, transformed_df, tgt)

    def test_to_spark(self):
        for tgt, datasets in self.tgt2datasets.items():
            transformer = Convert(astype="spark")
            go_products = datasets[3]
            self.assertEqual(get_table_name(go_products), "go_products", tgt)
            transformed_df = transformer.transform(go_products)
            self.assertTrue(_is_spark_df(transformed_df), tgt)
            self._check(go_products, transformed_df, tgt)

    def test_to_spark_with_index(self):
        for tgt, datasets in self.tgt2datasets.items():
            transformer = Convert(astype="spark-with-index")
            go_products = datasets[3]
            self.assertEqual(get_table_name(go_products), "go_products", tgt)
            transformed_df = transformer.transform(go_products)
            self.assertTrue(_is_spark_with_index(transformed_df), tgt)
            self._check(go_products, transformed_df, tgt)
