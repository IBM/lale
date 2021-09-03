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

import pandas as pd

from lale.datasets.data_schemas import add_table_name

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pyspark.sql import SparkSession

    spark_installed = True
except ImportError:
    spark_installed = False


def get_data_from_csv(datatype, data_file_name):
    if datatype.casefold() == "pandas":
        return pd.read_csv(data_file_name)
    elif datatype.casefold() == "spark":
        if spark_installed:
            spark = SparkSession.builder.appName("GoSales Dataset").getOrCreate()
            return spark.read.options(inferSchema="True", delimiter=",").csv(
                data_file_name, header=True
            )
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
      dataframe consisting of the data.

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
            urllib.request.urlretrieve(base_url + file, data_file_name)
            logger.info(" Created: {}".format(data_file_name))
        table_name = file.split(".")[0]
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
      table from the dataset) after reading the already existing CSV files.
      The key of each dictionary is the name of the table and the value contains
      a spark dataframe consisting of the data.

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
            "IMDB dataset not found at {}. Please download it using lalegpl repository.".format(
                download_data_dir
            )
        )
    else:
        for root, dirs, files in os.walk(download_data_dir):
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
                "Incomplete IMDB dataset found at {}. Please download complete dataset using lalegpl repository.".format(
                    download_data_dir
                )
            )
    return imdb_list
