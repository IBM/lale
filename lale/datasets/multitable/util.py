# Copyright 2019, 2020, 2021 IBM Corporation
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

from typing import Tuple

from numpy import random

try:
    from pyspark.sql import SparkSession

    spark_installed = True
except ImportError:
    spark_installed = False


from lale.datasets.data_schemas import add_table_name, get_table_name
from lale.helpers import _is_pandas_df, _is_spark_df


def multitable_train_test_split(
    dataset,
    main_table_name,
    label_column_name,
    test_size=0.25,
    random_state=None,
) -> Tuple:
    """
    Splits X and y into random train and test subsets stratified by
    labels and protected attributes.

    Behaves similar to the `train_test_split`_ function from scikit-learn.

    .. _`train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Parameters
    ----------
    dataset : list of either Pandas or Spark dataframes

      Each dataframe in the list corresponds to an entity/table in the multi-table setting.

    main_table_name : string

      The name of the main table as the split is going to be based on the main table.

    label_column_name : string

      The name of the label column from the main table.

    test_size : float or int, default=0.25

      If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
      If int, represents the absolute number of test samples.

    random_state : int, RandomState instance or None, default=None

      Controls the shuffling applied to the data before applying the split.
      Pass an integer for reproducible output across multiple function calls.

      - None

          RandomState used by numpy.random

      - numpy.random.RandomState

          Use the provided random state, only affecting other users of that same random state instance.

      - integer

          Explicit seed.

    Returns
    -------
    result : tuple

      - item 0: train_X, List of datasets corresponding to the train split

      - item 1: test_X, List of datasets corresponding to the test split

      - item 2: train_y

      - item 3: test_y

    """
    main_table_df = None
    index_of_main_table = -1
    for i, df in enumerate(dataset):
        if get_table_name(df) == main_table_name:
            main_table_df = df
            index_of_main_table = i
    if main_table_df is None:
        table_names = [get_table_name(df) for df in dataset]
        raise ValueError(
            f"Could not find {main_table_name} in the given dataset, the table names are {table_names}"
        )
    if _is_pandas_df(main_table_df):
        num_rows = len(main_table_df)
    elif _is_spark_df(main_table_df):
        # main_table_df = main_table_df.toPandas()
        num_rows = main_table_df.count()
    else:
        raise ValueError(
            "multitable_train_test_split can only work with a list of Pandas or Spark dataframes."
        )
    if test_size > 0 and test_size < 1:
        num_test_rows = int(num_rows * test_size)
    else:
        num_test_rows = test_size
    test_indices = random.choice(range(num_rows), num_test_rows, replace=False)
    train_indices = list(set([*range(num_rows)]) - set(test_indices.tolist()))
    assert len(test_indices) + len(train_indices) == num_rows
    train_dataset = [table for table in dataset]
    test_dataset = [table for table in dataset]
    if _is_pandas_df(main_table_df):
        train_main_df = main_table_df.iloc[train_indices]
        test_main_df = main_table_df.iloc[test_indices]
        train_y = train_main_df[label_column_name]
        test_y = test_main_df[label_column_name]
    elif _is_spark_df(main_table_df):
        spark_session = SparkSession.builder.appName(
            "multitable_train_test_split"
        ).getOrCreate()
        train_main_df = spark_session.createDataFrame(
            data=main_table_df.toPandas().iloc[train_indices]
        )
        test_main_df = spark_session.createDataFrame(
            data=main_table_df.toPandas().iloc[test_indices]
        )
        train_y = train_main_df.select(label_column_name)
        test_y = test_main_df.select(label_column_name)
    else:
        raise ValueError(
            "multitable_train_test_split can only work with a list of Pandas or Spark dataframes."
        )

    train_main_df = add_table_name(train_main_df, main_table_name)
    test_main_df = add_table_name(test_main_df, main_table_name)
    train_dataset[index_of_main_table] = train_main_df
    test_dataset[index_of_main_table] = test_main_df
    return train_dataset, test_dataset, train_y, test_y
