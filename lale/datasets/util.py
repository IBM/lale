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
import os
import pathlib

import pandas as pd

from lale.datasets.data_schemas import (
    SparkDataFrameWithIndex,
    add_table_name,
    get_table_name,
)

try:
    from pyspark.sql import SparkSession

except ImportError:
    SparkSession = None


download_data_cache_dir: pathlib.Path = pathlib.Path(
    os.environ.get("LALE_DOWNLOAD_CACHE_DIR", os.path.dirname(__file__))
)


def pandas2spark(pandas_df):
    assert SparkSession is not None

    spark_session = (
        SparkSession.builder.master("local[2]")  # type: ignore
        .config("spark.driver.memory", "64g")
        .getOrCreate()
    )
    name = get_table_name(pandas_df)
    if isinstance(pandas_df, pd.Series):
        pandas_df = pandas_df.to_frame()
    index_names = pandas_df.index.names
    if len(index_names) == 1 and index_names[0] is None:
        index_names = ["index"]
    cols = list(pandas_df.columns) + list(index_names)
    pandas_df = pandas_df.reset_index().reindex(columns=cols)
    spark_dataframe = spark_session.createDataFrame(pandas_df)
    spark_dataframe_with_index = SparkDataFrameWithIndex(spark_dataframe, index_names)
    return add_table_name(spark_dataframe_with_index, name)
