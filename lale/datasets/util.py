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
import sys
from typing import Any, Tuple, overload

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from lale.datasets.data_schemas import add_table_name, get_table_name

if sys.version_info >= (3, 8):
    from typing import Literal  # raises a mypy error for <3.8
else:
    from typing_extensions import Literal

try:
    from pyspark.sql import SparkSession

    from lale.datasets.data_schemas import (  # pylint:disable=ungrouped-imports
        SparkDataFrameWithIndex,
    )

    spark_installed = True
except ImportError:
    spark_installed = False


def pandas2spark(pandas_df):
    assert spark_installed
    spark_session = (
        SparkSession.builder.master("local[2]")
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


@overload
def load_boston(return_X_y: Literal[True]) -> Tuple[Any, Any]:
    ...


@overload
def load_boston(return_X_y: Literal[False] = False) -> Bunch:
    ...


def load_boston(return_X_y: bool = False):
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    if return_X_y:
        return (data, target)
    else:
        return Bunch(data=data, target=target)
