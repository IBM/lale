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

import re
import unittest

import jsonschema
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.lib.rasl import MinMaxScaler as RaslMinMaxScaler

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SQLContext

    spark_installed = True
except ImportError:
    spark_installed = False


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()

    def _check_trained(self, sk_trained, rasl_trained):
        self.assertEqual(list(sk_trained.data_min_), list(rasl_trained.impl.data_min_))
        self.assertEqual(list(sk_trained.data_max_), list(rasl_trained.impl.data_max_))
        self.assertEqual(
            list(sk_trained.data_range_), list(rasl_trained.impl.data_range_)
        )
        self.assertEqual(list(sk_trained.scale_), list(rasl_trained.impl.scale_))
        self.assertEqual(list(sk_trained.min_), list(rasl_trained.impl.min_))
        self.assertEqual(sk_trained.n_features_in_, rasl_trained.impl.n_features_in_)
        self.assertEqual(sk_trained.n_samples_seen_, rasl_trained.impl.n_samples_seen_)

    def test_get_params(self):
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_params = sk_scaler.get_params()
        rasl_params = rasl_scaler.get_params()
        self.assertDictContainsSubset(sk_params, rasl_params)

    def test_error(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError,
            re.compile(r"MinMaxScaler\(copy=False\)", re.MULTILINE | re.DOTALL),
        ):
            _ = RaslMinMaxScaler(copy=False)
        with self.assertRaisesRegex(
            jsonschema.ValidationError,
            re.compile(r"MinMaxScaler\(clip=True\)", re.MULTILINE | re.DOTALL),
        ):
            _ = RaslMinMaxScaler(clip=True)

    def test_fit(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data)
        self._check_trained(sk_trained, rasl_trained)

    def test_transform(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data)
        sk_transformed = sk_trained.transform(data)
        rasl_transformed = rasl_trained.transform(data)
        self.assertAlmostEqual(sk_transformed[0, 0], rasl_transformed.iloc[0, 0])
        self.assertAlmostEqual(sk_transformed[0, 1], rasl_transformed.iloc[0, 1])
        self.assertAlmostEqual(sk_transformed[0, 2], rasl_transformed.iloc[0, 2])
        self.assertAlmostEqual(sk_transformed[10, 0], rasl_transformed.iloc[10, 0])
        self.assertAlmostEqual(sk_transformed[10, 1], rasl_transformed.iloc[10, 1])
        self.assertAlmostEqual(sk_transformed[10, 2], rasl_transformed.iloc[10, 2])
        self.assertAlmostEqual(sk_transformed[20, 0], rasl_transformed.iloc[20, 0])
        self.assertAlmostEqual(sk_transformed[20, 1], rasl_transformed.iloc[20, 1])
        self.assertAlmostEqual(sk_transformed[20, 2], rasl_transformed.iloc[20, 2])

    def test_fit_range(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler(feature_range=(-5, 5))
        rasl_scaler = RaslMinMaxScaler(feature_range=(-5, 5))
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data)
        self._check_trained(sk_trained, rasl_trained)

    def test_transform_range(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler(feature_range=(-5, 5))
        rasl_scaler = RaslMinMaxScaler(feature_range=(-5, 5))
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data)
        sk_transformed = sk_trained.transform(data)
        rasl_transformed = rasl_trained.transform(data)
        self.assertAlmostEqual(sk_transformed[0, 0], rasl_transformed.iloc[0, 0])
        self.assertAlmostEqual(sk_transformed[0, 1], rasl_transformed.iloc[0, 1])
        self.assertAlmostEqual(sk_transformed[0, 2], rasl_transformed.iloc[0, 2])
        self.assertAlmostEqual(sk_transformed[10, 0], rasl_transformed.iloc[10, 0])
        self.assertAlmostEqual(sk_transformed[10, 1], rasl_transformed.iloc[10, 1])
        self.assertAlmostEqual(sk_transformed[10, 2], rasl_transformed.iloc[10, 2])
        self.assertAlmostEqual(sk_transformed[20, 0], rasl_transformed.iloc[20, 0])
        self.assertAlmostEqual(sk_transformed[20, 1], rasl_transformed.iloc[20, 1])
        self.assertAlmostEqual(sk_transformed[20, 2], rasl_transformed.iloc[20, 2])

    def test_partial_fit(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data1 = data[:10]
        data2 = data[10:100]
        data3 = data[100:]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.partial_fit(data1)
        rasl_trained = rasl_scaler.partial_fit(data1)
        self._check_trained(sk_trained, rasl_trained)
        sk_trained = sk_scaler.partial_fit(data2)
        rasl_trained = rasl_scaler.partial_fit(data2)
        self._check_trained(sk_trained, rasl_trained)
        sk_trained = sk_scaler.partial_fit(data3)
        rasl_trained = rasl_scaler.partial_fit(data3)
        self._check_trained(sk_trained, rasl_trained)


class TestMinMaxScalerSpark(unittest.TestCase):
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()
        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            self.sqlCtx = SQLContext(sc)

    def _check_trained(self, sk_trained, rasl_trained):
        self.assertEqual(list(sk_trained.data_min_), list(rasl_trained.impl.data_min_))
        self.assertEqual(list(sk_trained.data_max_), list(rasl_trained.impl.data_max_))
        self.assertEqual(
            list(sk_trained.data_range_), list(rasl_trained.impl.data_range_)
        )
        self.assertEqual(list(sk_trained.scale_), list(rasl_trained.impl.scale_))
        self.assertEqual(list(sk_trained.min_), list(rasl_trained.impl.min_))
        self.assertEqual(sk_trained.n_features_in_, rasl_trained.impl.n_features_in_)
        self.assertEqual(sk_trained.n_samples_seen_, rasl_trained.impl.n_samples_seen_)

    def test_fit(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data_spark = self.sqlCtx.createDataFrame(data)
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data_spark)
        self._check_trained(sk_trained, rasl_trained)

    def test_transform(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data_spark = self.sqlCtx.createDataFrame(data)
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data_spark)
        sk_transformed = sk_trained.transform(data)
        rasl_transformed = rasl_trained.transform(data_spark)
        rasl_transformed = rasl_transformed.toPandas()
        self.assertAlmostEqual(sk_transformed[0, 0], rasl_transformed.iloc[0, 0])
        self.assertAlmostEqual(sk_transformed[0, 1], rasl_transformed.iloc[0, 1])
        self.assertAlmostEqual(sk_transformed[0, 2], rasl_transformed.iloc[0, 2])
        self.assertAlmostEqual(sk_transformed[10, 0], rasl_transformed.iloc[10, 0])
        self.assertAlmostEqual(sk_transformed[10, 1], rasl_transformed.iloc[10, 1])
        self.assertAlmostEqual(sk_transformed[10, 2], rasl_transformed.iloc[10, 2])
        self.assertAlmostEqual(sk_transformed[20, 0], rasl_transformed.iloc[20, 0])
        self.assertAlmostEqual(sk_transformed[20, 1], rasl_transformed.iloc[20, 1])
        self.assertAlmostEqual(sk_transformed[20, 2], rasl_transformed.iloc[20, 2])

    def test_fit_range(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data_spark = self.sqlCtx.createDataFrame(data)
        sk_scaler = SkMinMaxScaler(feature_range=(-5, 5))
        rasl_scaler = RaslMinMaxScaler(feature_range=(-5, 5))
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data_spark)
        self._check_trained(sk_trained, rasl_trained)

    def test_transform_range(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data_spark = self.sqlCtx.createDataFrame(data)
        sk_scaler = SkMinMaxScaler(feature_range=(-5, 5))
        rasl_scaler = RaslMinMaxScaler(feature_range=(-5, 5))
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data_spark)
        sk_transformed = sk_trained.transform(data)
        rasl_transformed = rasl_trained.transform(data_spark)
        rasl_transformed = rasl_transformed.toPandas()
        self.assertAlmostEqual(sk_transformed[0, 0], rasl_transformed.iloc[0, 0])
        self.assertAlmostEqual(sk_transformed[0, 1], rasl_transformed.iloc[0, 1])
        self.assertAlmostEqual(sk_transformed[0, 2], rasl_transformed.iloc[0, 2])
        self.assertAlmostEqual(sk_transformed[10, 0], rasl_transformed.iloc[10, 0])
        self.assertAlmostEqual(sk_transformed[10, 1], rasl_transformed.iloc[10, 1])
        self.assertAlmostEqual(sk_transformed[10, 2], rasl_transformed.iloc[10, 2])
        self.assertAlmostEqual(sk_transformed[20, 0], rasl_transformed.iloc[20, 0])
        self.assertAlmostEqual(sk_transformed[20, 1], rasl_transformed.iloc[20, 1])
        self.assertAlmostEqual(sk_transformed[20, 2], rasl_transformed.iloc[20, 2])

    def test_partial_fit(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data1 = data[:10]
        data1_spark = self.sqlCtx.createDataFrame(data1)
        data2 = data[10:100]
        data2_spark = self.sqlCtx.createDataFrame(data2)
        data3 = data[100:]
        data3_spark = self.sqlCtx.createDataFrame(data3)
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.partial_fit(data1)
        rasl_trained = rasl_scaler.partial_fit(data1_spark)
        self._check_trained(sk_trained, rasl_trained)
        sk_trained = sk_scaler.partial_fit(data2)
        rasl_trained = rasl_scaler.partial_fit(data2_spark)
        self._check_trained(sk_trained, rasl_trained)
        sk_trained = sk_scaler.partial_fit(data3)
        rasl_trained = rasl_scaler.partial_fit(data3_spark)
        self._check_trained(sk_trained, rasl_trained)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris(as_frame=True)
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

        if spark_installed:
            conf = (
                SparkConf()
                .setMaster("local[2]")
                .set("spark.driver.bindAddress", "127.0.0.1")
            )
            sc = SparkContext.getOrCreate(conf=conf)
            self.sqlCtx = SQLContext(sc)
            self.X_train_spark = self.sqlCtx.createDataFrame(self.X_train)
            self.X_test_spark = self.sqlCtx.createDataFrame(self.X_test)

    def test_pipeline_pandas(self):
        from lale.lib.sklearn import LogisticRegression

        pipeline = RaslMinMaxScaler() >> LogisticRegression()
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    def test_pipeline_spark(self):
        from lale.lib.sklearn import FunctionTransformer, LogisticRegression

        if spark_installed:
            pipeline = (
                RaslMinMaxScaler()
                >> FunctionTransformer(func=lambda X: X.toPandas())
                >> LogisticRegression()
            )
            trained = pipeline.fit(self.X_train_spark, self.y_train)
            _ = trained.predict(self.X_test_spark)
