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

from lale.datasets.data_schemas import add_table_name, get_table_name

try:
    import pyspark.sql
    from pyspark import SparkConf, SparkContext

    from lale.datasets.data_schemas import SparkDataFrameWithIndex

    spark_installed = True
except ImportError:
    spark_installed = False


def pandas2spark(pandas_df, with_index=False):
    assert spark_installed
    spark_conf = (
        SparkConf().setMaster("local[2]").set("spark.driver.bindAddress", "127.0.0.1")
    )
    spark_context = SparkContext.getOrCreate(conf=spark_conf)
    spark_sql_context = pyspark.sql.SQLContext(spark_context)
    name = get_table_name(pandas_df)
    if isinstance(pandas_df, pd.Series):
        pandas_df = pandas_df.to_frame()
    index_names = None
    if with_index:
        index_names = pandas_df.index.names
        if len(index_names) == 1 and index_names[0] is None:
            index_names = ["index"]
        cols = list(pandas_df.columns) + list(index_names)
        pandas_df = pandas_df.reset_index().reindex(columns=cols)
    spark_dataframe = spark_sql_context.createDataFrame(pandas_df)
    if index_names is not None:
        spark_dataframe = SparkDataFrameWithIndex(spark_dataframe, index_names)
    return add_table_name(spark_dataframe, name)
