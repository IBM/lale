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
    from pyspark.sql import SQLContext

    spark_installed = True
except ImportError:
    spark_installed = False

from test import EnableSchemaValidation

from lale import wrap_imported_operators
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
    Hyperopt,
    Join,
    Map,
    Relational,
    Scan,
)
from lale.lib.sklearn import KNeighborsClassifier, LogisticRegression
from lale.operators import make_pipeline_graph


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
