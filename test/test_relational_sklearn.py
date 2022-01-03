# Copyright 2021 IBM Corporation
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
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from sklearn.feature_selection import SelectKBest as SkSelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

import lale.datasets
import lale.datasets.openml
from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.expressions import it
from lale.lib.lale import Scan, categorical
from lale.lib.rasl import Map
from lale.lib.rasl import MinMaxScaler as RaslMinMaxScaler
from lale.lib.rasl import OneHotEncoder as RaslOneHotEncoder
from lale.lib.rasl import OrdinalEncoder as RaslOrdinalEncoder
from lale.lib.sklearn import FunctionTransformer, LogisticRegression


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
        data_spark = lale.datasets.pandas2spark(data)
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data_spark)
        self._check_trained(sk_trained, rasl_trained)

    def test_transform(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data_spark = lale.datasets.pandas2spark(data)
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
        data_spark = lale.datasets.pandas2spark(data)
        sk_scaler = SkMinMaxScaler(feature_range=(-5, 5))
        rasl_scaler = RaslMinMaxScaler(feature_range=(-5, 5))
        sk_trained = sk_scaler.fit(data)
        rasl_trained = rasl_scaler.fit(data_spark)
        self._check_trained(sk_trained, rasl_trained)

    def test_transform_range(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        data_spark = lale.datasets.pandas2spark(data)
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
        data1_spark = lale.datasets.pandas2spark(data1)
        data2 = data[10:100]
        data2_spark = lale.datasets.pandas2spark(data2)
        data3 = data[100:]
        data3_spark = lale.datasets.pandas2spark(data3)
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

        X, y = load_iris(as_frame=True, return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.X_train_spark = lale.datasets.pandas2spark(self.X_train)
        self.X_test_spark = lale.datasets.pandas2spark(self.X_test)

    def test_pipeline_pandas(self):
        pipeline = RaslMinMaxScaler() >> LogisticRegression()
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    def test_pipeline_spark(self):
        pipeline = (
            RaslMinMaxScaler()
            >> FunctionTransformer(func=lambda X: X.toPandas())
            >> LogisticRegression()
        )
        trained = pipeline.fit(self.X_train_spark, self.y_train)
        _ = trained.predict(self.X_test_spark)

class TestSelectKBest(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_digits
        self.X, self.y = load_digits(return_X_y=True)

    def _check_trained(self, sk_trained, rasl_trained):
        np.testing.assert_equal(sk_trained.scores_, rasl_trained.scores_)
        np.testing.assert_equal(sk_trained.pvalues_, rasl_trained.pvalues_)
        self.assertEqual(sk_trained.n_features_in_, rasl_trained.n_features_in_)

    def test_fit(self):
        sk_trainable = SkSelectKBest(chi2, k=20)
        sk_trained = sk_trainable.fit(self.X, self.y)
        self._check_trained(sk_trained, sk_trained)

class TestOrdinalEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark"]
        cls.tgt2gosales = {tgt: fetch_go_sales_dataset(tgt) for tgt in targets}
        cls.tgt2creditg = {
            tgt: lale.datasets.openml.fetch(
                "credit-g",
                "classification",
                preprocess=False,
                astype=tgt,
            )
            for tgt in targets
        }

    def test_fit(self):
        prefix = Scan(table=it.go_daily_sales) >> Map(
            columns={"retailer": it["Retailer code"], "method": it["Order method code"]}
        )
        encoder_args = {"handle_unknown": "use_encoded_value", "unknown_value": np.nan}
        rasl_trainable = prefix >> RaslOrdinalEncoder(**encoder_args)
        sk_trainable = prefix >> SkOrdinalEncoder(**encoder_args)
        sk_trained = sk_trainable.fit(self.tgt2gosales["pandas"])
        sk_categories = sk_trained.get_last().impl.categories_
        for tgt, datasets in self.tgt2gosales.items():
            rasl_trained = rasl_trainable.fit(datasets)
            rasl_categories = rasl_trained.get_last().impl.categories_
            self.assertEqual(len(sk_categories), len(rasl_categories), tgt)
            for i in range(len(sk_categories)):
                self.assertEqual(list(sk_categories[i]), list(rasl_categories[i]), tgt)

    def test_transform(self):
        prefix = Scan(table=it.go_daily_sales) >> Map(
            columns={"retailer": it["Retailer code"], "method": it["Order method code"]}
        )
        encoder_args = {"handle_unknown": "use_encoded_value", "unknown_value": np.nan}
        rasl_trainable = prefix >> RaslOrdinalEncoder(**encoder_args)
        sk_trainable = prefix >> SkOrdinalEncoder(**encoder_args)
        sk_trained = sk_trainable.fit(self.tgt2gosales["pandas"])
        sk_transformed = sk_trained.transform(self.tgt2gosales["pandas"])
        for tgt, datasets in self.tgt2gosales.items():
            rasl_trained = rasl_trainable.fit(datasets)
            rasl_transformed = rasl_trained.transform(datasets)
            if tgt == "spark":
                rasl_transformed = rasl_transformed.toPandas()
            self.assertEqual(sk_transformed.shape, rasl_transformed.shape, tgt)
            for row_idx in range(sk_transformed.shape[0]):
                for col_idx in range(sk_transformed.shape[1]):
                    self.assertEqual(
                        sk_transformed[row_idx, col_idx],
                        rasl_transformed.iloc[row_idx, col_idx],
                        (row_idx, col_idx, tgt),
                    )

    def test_predict(self):
        (train_X_pd, train_y_pd), (test_X_pd, test_y_pd) = self.tgt2creditg["pandas"]
        cat_columns = categorical()(train_X_pd)
        prefix = Map(columns={c: it[c] for c in cat_columns})
        to_pd = FunctionTransformer(
            func=lambda X: X if isinstance(X, pd.DataFrame) else X.toPandas()
        )
        lr = LogisticRegression()
        encoder_args = {"handle_unknown": "use_encoded_value", "unknown_value": -1}
        sk_trainable = prefix >> SkOrdinalEncoder(**encoder_args) >> lr
        sk_trained = sk_trainable.fit(train_X_pd, train_y_pd)
        sk_predicted = sk_trained.predict(test_X_pd)
        rasl_trainable = prefix >> RaslOrdinalEncoder(**encoder_args) >> to_pd >> lr
        for tgt, dataset in self.tgt2creditg.items():
            (train_X, train_y), (test_X, test_y) = dataset
            rasl_trained = rasl_trainable.fit(train_X, train_y)
            rasl_predicted = rasl_trained.predict(test_X)
            self.assertEqual(sk_predicted.shape, rasl_predicted.shape, tgt)
            self.assertEqual(sk_predicted.tolist(), rasl_predicted.tolist(), tgt)


class TestOneHotEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = ["pandas", "spark"]
        cls.tgt2creditg = {
            tgt: lale.datasets.openml.fetch(
                "credit-g",
                "classification",
                preprocess=False,
                astype=tgt,
            )
            for tgt in targets
        }

    def test_fit(self):
        (train_X_pd, train_y_pd), (test_X_pd, test_y_pd) = self.tgt2creditg["pandas"]
        cat_columns = categorical()(train_X_pd)
        prefix = Map(columns={c: it[c] for c in cat_columns})
        rasl_trainable = prefix >> RaslOneHotEncoder()
        sk_trainable = prefix >> SkOneHotEncoder()
        sk_trained = sk_trainable.fit(train_X_pd)
        sk_categories = sk_trained.get_last().impl.categories_
        for tgt, dataset in self.tgt2creditg.items():
            (train_X, train_y), (test_X, test_y) = dataset
            rasl_trained = rasl_trainable.fit(train_X)
            rasl_categories = rasl_trained.get_last().impl.categories_
            self.assertEqual(len(sk_categories), len(rasl_categories), tgt)
            for i in range(len(sk_categories)):
                self.assertEqual(list(sk_categories[i]), list(rasl_categories[i]), tgt)

    def test_transform(self):
        (train_X_pd, train_y_pd), (test_X_pd, test_y_pd) = self.tgt2creditg["pandas"]
        cat_columns = categorical()(train_X_pd)
        prefix = Map(columns={c: it[c] for c in cat_columns})
        rasl_trainable = prefix >> RaslOneHotEncoder(sparse=False)
        sk_trainable = prefix >> SkOneHotEncoder(sparse=False)
        sk_trained = sk_trainable.fit(train_X_pd)
        sk_transformed = sk_trained.transform(test_X_pd)
        for tgt, dataset in self.tgt2creditg.items():
            (train_X, train_y), (test_X, test_y) = dataset
            rasl_trained = rasl_trainable.fit(train_X)
            rasl_transformed = rasl_trained.transform(test_X)
            if tgt == "spark":
                rasl_transformed = rasl_transformed.toPandas()
            self.assertEqual(sk_transformed.shape, rasl_transformed.shape, tgt)
            for row_idx in range(sk_transformed.shape[0]):
                for col_idx in range(sk_transformed.shape[1]):
                    self.assertEqual(
                        sk_transformed[row_idx, col_idx],
                        rasl_transformed.iloc[row_idx, col_idx],
                        (row_idx, col_idx, tgt),
                    )

    def test_predict(self):
        (train_X_pd, train_y_pd), (test_X_pd, test_y_pd) = self.tgt2creditg["pandas"]
        cat_columns = categorical()(train_X_pd)
        prefix = Map(columns={c: it[c] for c in cat_columns})
        to_pd = FunctionTransformer(
            func=lambda X: X if isinstance(X, pd.DataFrame) else X.toPandas()
        )
        lr = LogisticRegression()
        sk_trainable = prefix >> SkOneHotEncoder(sparse=False) >> lr
        sk_trained = sk_trainable.fit(train_X_pd, train_y_pd)
        sk_predicted = sk_trained.predict(test_X_pd)
        rasl_trainable = prefix >> RaslOneHotEncoder(sparse=False) >> to_pd >> lr
        for tgt, dataset in self.tgt2creditg.items():
            (train_X, train_y), (test_X, test_y) = dataset
            rasl_trained = rasl_trainable.fit(train_X, train_y)
            rasl_predicted = rasl_trained.predict(test_X)
            self.assertEqual(sk_predicted.shape, rasl_predicted.shape, tgt)
            self.assertEqual(sk_predicted.tolist(), rasl_predicted.tolist(), tgt)
