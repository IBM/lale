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

import logging
import os
import urllib.request

import numpy as np
import pandas as pd

import lale.datasets.openml
from lale.datasets.data_schemas import add_table_name

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pyspark.sql import SparkSession

    from lale.datasets.data_schemas import (  # pylint:disable=ungrouped-imports
        SparkDataFrameWithIndex,
    )

    spark_installed = True
except ImportError:
    spark_installed = False


def get_data_from_csv(datatype, data_file_name):
    datatype = datatype.casefold()
    if datatype == "pandas":
        return pd.read_csv(data_file_name)
    elif datatype == "spark":
        if spark_installed:
            spark = SparkSession.builder.appName("GoSales Dataset").getOrCreate()
            df = spark.read.options(inferSchema="True", delimiter=",").csv(
                data_file_name, header=True
            )
            return SparkDataFrameWithIndex(df)
        else:
            raise ValueError("Spark is not installed on this machine.")
    else:
        raise ValueError(
            "Can fetch the go_sales data in pandas or spark dataframes only. Pass either 'pandas' or 'spark' in datatype parameter."
        )


def fetch_go_sales_dataset(datatype="pandas"):

    """
    Fetches the Go_Sales dataset from IBM's Watson's ML samples.
    It contains information about daily sales, methods, retailers
    and products of a company in form of 5 CSV files.
    This method downloads and stores these 5 CSV files under the
    'lale/lale/datasets/multitable/go_sales_data' directory. It creates
    this directory by itself if it does not exists.

    Dataset URL: https://github.com/IBM/watson-machine-learning-samples/raw/master/cloud/data/go_sales/

    Parameters
    ----------
    datatype : string, optional, default 'pandas'

      If 'pandas',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the downloaded CSV files. The key of
      each dictionary is the name of the table and the value contains a pandas
      dataframe consisting of the data.

      If 'spark',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the downloaded CSV files. The key of
      each dictionary is the name of the table and the value contains a spark
      dataframe consisting of the data extended with an index column.

      Else,
      Throws an error as it does not support any other return type.

    Returns
    -------
    go_sales_list : list of singleton dictionary of pandas / spark dataframes
    """

    download_data_dir = os.path.join(os.path.dirname(__file__), "go_sales_data")
    base_url = "https://github.com/IBM/watson-machine-learning-samples/raw/master/cloud/data/go_sales/"
    filenames = [
        "go_1k.csv",
        "go_daily_sales.csv",
        "go_methods.csv",
        "go_products.csv",
        "go_retailers.csv",
    ]
    go_sales_list = []
    for file in filenames:
        data_file_name = os.path.join(download_data_dir, file)
        if not os.path.exists(data_file_name):
            if not os.path.exists(download_data_dir):
                os.makedirs(download_data_dir)
            # this request is to a hardcoded https url, so does not risk leaking local data
            urllib.request.urlretrieve(base_url + file, data_file_name)  # nosec
            logger.info(f" Created: {data_file_name}")
        table_name = file.split(".", maxsplit=1)[0]
        data_frame = get_data_from_csv(datatype, data_file_name)
        go_sales_list.append(add_table_name(data_frame, table_name))
    logger.info(" Fetched the Go_Sales dataset. Process completed.")
    return go_sales_list


def fetch_imdb_dataset(datatype="pandas"):

    """
    Fetches the IMDB movie dataset from Relational Dataset Repo.
    It contains information about directors, actors, roles
    and genres of multiple movies in form of 7 CSV files.
    This method downloads and stores these 7 CSV files under the
    'lale/lale/datasets/multitable/imdb_data' directory. It creates
    this directory by itself if it does not exists.

    Dataset URL: https://relational.fit.cvut.cz/dataset/IMDb

    Parameters
    ----------
    datatype : string, optional, default 'pandas'

      If 'pandas',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the already existing CSV files.
      The key of each dictionary is the name of the table and the value contains
      a pandas dataframe consisting of the data.

      If 'spark',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the downloaded CSV files. The key of
      each dictionary is the name of the table and the value contains a spark
      dataframe consisting of the data extended with an index column.

      Else,
      Throws an error as it does not support any other return type.

    Returns
    -------
    imdb_list : list of singleton dictionary of pandas / spark dataframes
    """

    download_data_dir = os.path.join(os.path.dirname(__file__), "imdb_data")
    imdb_list = []
    if not os.path.exists(download_data_dir):
        raise ValueError(
            f"IMDB dataset not found at {download_data_dir}. Please download it using lalegpl repository."
        )

    for _root, _dirs, files in os.walk(download_data_dir):
        for file in files:
            filename, extension = os.path.splitext(file)
            if extension == ".csv":
                data_file_name = os.path.join(download_data_dir, file)
                table_name = filename
                data_frame = get_data_from_csv(datatype, data_file_name)
                imdb_list.append(add_table_name(data_frame, table_name))
    if len(imdb_list) == 7:
        logger.info(" Fetched the IMDB dataset. Process completed.")
    else:
        raise ValueError(
            f"Incomplete IMDB dataset found at {download_data_dir}. Please download complete dataset using lalegpl repository."
        )
    return imdb_list


def fetch_creditg_multitable_dataset(datatype="pandas"):

    """
    Fetches credit-g dataset from OpenML, but in a multi-table format.
    It transforms the [credit-g](https://www.openml.org/d/31) dataset from OpenML
    to a multi-table format. We split the dataset into 3 tables: `loan_application`,
    `bank_account_info` and `existing_credits_info`.
    The table `loan_application` serves as our primary table,
    and we treat the other two tables as providing additional information related to
    the applicant's bank account and existing credits. As one can see, this is very
    close to a real life scenario where information is present in multiple tables in
    normalized forms. We created a primary key column `id` as a proxy to the loan applicant's
    identity number.

    Parameters
    ----------
    datatype : string, optional, default 'pandas'

      If 'pandas',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the downloaded CSV files. The key of
      each dictionary is the name of the table and the value contains a pandas
      dataframe consisting of the data.

    Returns
    -------
    dataframes_list : list of singleton dictionary of pandas dataframes
    """
    (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(
        "credit-g", "classification", preprocess=False
    )
    # vstack train and test
    X = pd.concat([train_X, test_X], axis=0)
    y = pd.concat([train_y, test_y], axis=0)

    bank_account_columns = ["checking_status", "savings_status"]
    loan_application_columns = [
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "employment",
        "installment_commitment",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
    ]
    dataframes_list = []

    bank_acc_df = X[bank_account_columns]
    bank_acc_df = bank_acc_df.copy()
    bank_acc_df.insert(0, "id", bank_acc_df.index)
    dataframes_list.append(add_table_name(bank_acc_df, "bank_account_info"))

    loan_application_df = X[loan_application_columns]
    loan_application_df = loan_application_df.copy()
    loan_application_df.insert(0, "id", loan_application_df.index)
    loan_application_df["class"] = y
    loan_application_df.iloc[2, 7] = "M single"
    loan_application_df.iloc[996, 7] = "M single"
    loan_application_df.iloc[998, 7] = "F div/dep/mar"
    dataframes_list.append(add_table_name(loan_application_df, "loan_application"))

    # existing credits is a fake table we are adding, so a join and count can create the `existing_credits` column
    df_col = X["existing_credits"]
    records = []
    for row in df_col.iteritems():
        row_id = row[0]
        credit_count = int(row[1])
        for _i in range(credit_count):
            records.append(
                {
                    "id": row_id,
                    "credit_number": np.random.randint(1, 1000000),
                    "type": "credit",
                    "status": "on",
                }
            )
    existing_credits_df = pd.DataFrame.from_records(records)
    dataframes_list.append(add_table_name(existing_credits_df, "existing_credits_info"))
    return dataframes_list
