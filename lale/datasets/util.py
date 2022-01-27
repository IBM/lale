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

import pandas as pd

from lale.datasets.data_schemas import SparkDataFrameWithIndex

try:
    import pyspark.sql
    from pyspark import SparkConf, SparkContext

    spark_installed = True
except ImportError:
    spark_installed = False


def pandas2spark(pandas_dataframe, add_index=False, index_name=None):
    assert spark_installed
    spark_conf = (
        SparkConf().setMaster("local[2]").set("spark.driver.bindAddress", "127.0.0.1")
    )
    spark_context = SparkContext.getOrCreate(conf=spark_conf)
    spark_sql_context = pyspark.sql.SQLContext(spark_context)
    if isinstance(pandas_dataframe, pd.Series):
        pandas_dataframe = pandas_dataframe.to_frame()
    if add_index:
        if index_name is None:
            if pandas_dataframe.index.name is None:
                index_name = "index"
            else:
                index_name = pandas_dataframe.index.name
        pandas_dataframe = pandas_dataframe.copy(deep=False)
        pandas_dataframe[index_name] = pandas_dataframe.index

    spark_dataframe = spark_sql_context.createDataFrame(pandas_dataframe)
    if index_name is not None:
        spark_dataframe = SparkDataFrameWithIndex(spark_dataframe)
    return spark_dataframe
