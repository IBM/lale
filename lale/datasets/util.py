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

try:
    import pyspark.sql
    from pyspark import SparkConf, SparkContext

    spark_installed = True
except ImportError:
    spark_installed = False


def pandas2spark(pandas_dataframe):
    assert spark_installed
    spark_conf = (
        SparkConf().setMaster("local[2]").set("spark.driver.bindAddress", "127.0.0.1")
    )
    spark_context = SparkContext.getOrCreate(conf=spark_conf)
    spark_sql_context = pyspark.sql.SQLContext(spark_context)
    spark_dataframe = spark_sql_context.createDataFrame(pandas_dataframe)
    return spark_dataframe
