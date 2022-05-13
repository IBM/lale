# Copyright 2022 IBM Corporation
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

from typing import Iterable, Tuple, Union, cast

import pandas as pd
import sklearn.model_selection
import sklearn.tree

import lale.helpers
from lale.datasets import pandas2spark

from .split_xy import SplitXy

_PandasBatch = Tuple[pd.DataFrame, pd.Series]

if lale.helpers.spark_installed:
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame

    _PandasOrSparkBatch = Union[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[SparkDataFrame, SparkDataFrame],
    ]
else:
    _PandasOrSparkBatch = _PandasBatch  # type: ignore

try:
    import arff

    from lale.datasets.openml import openml_datasets

    liac_arff_installed = True
except ModuleNotFoundError:
    liac_arff_installed = False


def arff_data_loader(
    file_name: str, label_name: str, rows_per_batch: int
) -> Iterable[_PandasBatch]:
    """Incrementally load an ARFF file and yield it one (X, y) batch at a time."""
    assert liac_arff_installed
    split_x_y = SplitXy(label_name=label_name)

    def make_batch():
        start = n_batches * rows_per_batch
        stop = start + len(row_list)
        df = pd.DataFrame(row_list, range(start, stop), column_names)
        X, y = split_x_y.transform_X_y(df, None)
        return X, y

    with open(file_name) as f:
        arff_dict = arff.load(f, return_type=arff.DENSE_GEN)
        column_names = [name.lower() for name, _ in arff_dict["attributes"]]
        row_list = []
        n_batches = 0
        for row in arff_dict["data"]:
            row_list.append(row)
            if len(row_list) >= rows_per_batch:
                yield make_batch()
                row_list = []
                n_batches += 1
    if len(row_list) > 0:  # last chunk
        yield make_batch()


def csv_data_loader(
    file_name: str, label_name: str, rows_per_batch: int
) -> Iterable[_PandasBatch]:
    """Incrementally load an CSV file and yield it one (X, y) batch at a time."""
    split_x_y = SplitXy(label_name=label_name)
    with pd.read_csv(file_name, chunksize=rows_per_batch) as reader:
        for df in reader:
            X, y = split_x_y.transform_X_y(df, None)
            yield X, y


def mockup_data_loader(
    X: pd.DataFrame, y: pd.Series, n_batches: int, astype: str
) -> Iterable[_PandasOrSparkBatch]:
    """Split (X, y) into batches to emulate loading them incrementally.

    Only intended for testing purposes, because if X and y are already
    materialized in-memory, there is little reason to batch them.
    """
    pandas_gen: Iterable[_PandasBatch]
    if n_batches == 1:
        pandas_gen = [(X, y)]
    else:
        cv = sklearn.model_selection.KFold(n_batches)
        estimator = sklearn.tree.DecisionTreeClassifier()
        pandas_gen = (
            lale.helpers.split_with_schemas(estimator, X, y, test, train)
            for train, test in cv.split(X, y)
        )
    if astype == "pandas":
        return pandas_gen
    elif astype == "spark":
        return ((pandas2spark(X), pandas2spark(y)) for X, y in pandas_gen)
    raise ValueError(f"expected astype in ['pandas', 'spark'], got {astype}")


def openml_data_loader(dataset_name: str, batch_size: int) -> Iterable[_PandasBatch]:
    """Download the OpenML dataset, incrementally load it, and yield it one (X,y) batch at a time."""
    assert liac_arff_installed
    metadata = openml_datasets.experiments_dict[dataset_name]
    label_name = cast(str, metadata["target"]).lower()
    file_name = openml_datasets.download_if_missing(dataset_name)
    return arff_data_loader(file_name, label_name, batch_size)
