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

from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.lib.rasl import MinMaxScaler as RaslMinMaxScaler


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()
        # self.go_sales_spark = fetch_go_sales_dataset("spark")

    def test_fit(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trainned = sk_scaler.fit(data)
        rasl_trainned = rasl_scaler.fit(data)
        self.assertTrue((sk_trainned.data_min_ == rasl_trainned.impl.data_min_).all())
        self.assertTrue((sk_trainned.data_max_ == rasl_trainned.impl.data_max_).all())
        self.assertTrue(
            (sk_trainned.data_range_ == rasl_trainned.impl.data_range_).all()
        )
        self.assertEqual(sk_trainned.n_features_in_, rasl_trainned.impl.n_features_in_)
        # self.assertEqual(sk_trainned.feature_names_in_, rasl_trainned.impl.feature_names_in_)

    def test_transform(self):
        columns = ["Product number", "Quantity", "Retailer code"]
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trainned = sk_scaler.fit(data)
        rasl_trainned = rasl_scaler.fit(data)
        sk_transformed = sk_trainned.transform(data)
        rasl_transformed = rasl_trainned.transform(data)
        self.assertAlmostEqual(sk_transformed[0, 0], rasl_transformed.iloc[0, 0])
        self.assertAlmostEqual(sk_transformed[0, 1], rasl_transformed.iloc[0, 1])
        self.assertAlmostEqual(sk_transformed[0, 2], rasl_transformed.iloc[0, 2])
        self.assertAlmostEqual(sk_transformed[10, 0], rasl_transformed.iloc[10, 0])
        self.assertAlmostEqual(sk_transformed[10, 1], rasl_transformed.iloc[10, 1])
        self.assertAlmostEqual(sk_transformed[10, 2], rasl_transformed.iloc[10, 2])
        self.assertAlmostEqual(sk_transformed[20, 0], rasl_transformed.iloc[20, 0])
        self.assertAlmostEqual(sk_transformed[20, 1], rasl_transformed.iloc[20, 1])
        self.assertAlmostEqual(sk_transformed[20, 2], rasl_transformed.iloc[20, 2])
