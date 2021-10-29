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

from lale import wrap_imported_operators
from lale.datasets.data_schemas import add_table_name, get_table_name
from lale.datasets.multitable import multitable_train_test_split
from lale.datasets.multitable.fetch_datasets import fetch_go_sales_dataset
from lale.expressions import (
    asc,
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
    ratio,
    replace,
    string_indexer,
    subtract,
    sum,
)
from lale.helpers import _is_pandas_df, _is_spark_df
from lale.lib.lale import (
    Aggregate,
    Alias,
    ConcatFeatures,
    Filter,
    GroupBy,
    Hyperopt,
    Join,
    Map,
    OrderBy,
    Relational,
    Scan,
    SplitXy,
)
from lale.lib.sklearn import PCA, KNeighborsClassifier, LogisticRegression
from lale.operators import make_pipeline_graph

from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from lale.lib.rasl import MinMaxScaler as RaslMinMaxScaler

class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.go_sales = fetch_go_sales_dataset()
        # self.go_sales_spark = fetch_go_sales_dataset("spark")

    def test_fit(self):
        columns = ['Product number', 'Quantity', 'Retailer code']
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trainned = sk_scaler.fit(data)
        rasl_trainned = rasl_scaler.fit(data)
        self.assertTrue((sk_trainned.data_min_ == rasl_trainned.impl.data_min_).all())
        self.assertTrue((sk_trainned.data_max_ == rasl_trainned.impl.data_max_).all())
        self.assertTrue((sk_trainned.data_range_ == rasl_trainned.impl.data_range_).all())
        self.assertEqual(sk_trainned.n_features_in_, rasl_trainned.impl.n_features_in_)
        # self.assertEqual(sk_trainned.feature_names_in_, rasl_trainned.impl.feature_names_in_)

    def test_transform(self):
        columns = ['Product number', 'Quantity', 'Retailer code']
        data = self.go_sales[0][columns]
        sk_scaler = SkMinMaxScaler()
        rasl_scaler = RaslMinMaxScaler()
        sk_trainned = sk_scaler.fit(data)
        rasl_trainned = rasl_scaler.fit(data)
        sk_transformed = sk_trainned.transform(data)
        rasl_transformed = rasl_trainned.transform(data)
        print('XXXXXXXX')
        print(sk_transformed)
        print(rasl_transformed)
