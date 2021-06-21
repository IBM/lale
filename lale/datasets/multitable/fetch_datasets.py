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

import os, urllib.request, sys, csv
import pandas as pd
import mysql.connector

try:
    from pyspark.sql import SparkSession

    spark_installed = True
except ImportError:
    spark_installed = False

imdb_config = {
  'user': 'guest',
  'password': 'relational',
  'host': 'relational.fit.cvut.cz',
  'database': 'imdb_ijs',
  'port': 3306,
  'raise_on_warnings': True
}

def get_data_from_csv(datatype, data_file_name):
    if datatype.casefold() == "pandas":
         return pd.read_csv(data_file_name)
    elif datatype.casefold() == "spark":
        if spark_installed:
            spark = SparkSession.builder.appName("GoSales Dataset").getOrCreate()
            return spark.read.csv(data_file_name, header=True)
        else:
            raise ValueError("Spark is not installed on this machine!")
    else:
        raise ValueError("Can fetch the go_sales data in pandas or spark dataframes only! Pass either 'pandas' or 'spark' in datatype parameter!")


def fetch_go_sales_dataset(datatype='pandas'):
    download_data_dir = os.path.join(os.path.dirname(__file__), "go_sales_data")
    base_url = 'https://github.com/IBM/watson-machine-learning-samples/raw/master/cloud/data/go_sales/'
    filenames = ['go_1k.csv', 'go_daily_sales.csv', 'go_methods.csv', 'go_products.csv', 'go_retailers.csv']
    go_sales_dict = {}
    for file in filenames:
        data_file_name = os.path.join(download_data_dir, file)
        if not os.path.exists(data_file_name):
            if not os.path.exists(download_data_dir):
                os.makedirs(download_data_dir)
            urllib.request.urlretrieve(base_url + file, data_file_name)
            print("Created:", file)
        go_sales_dict[file.split('.')[0]] = get_data_from_csv(datatype, data_file_name)
    return go_sales_dict

def fetch_imdb_dataset(datatype='pandas'):
    try:
        cnx = mysql.connector.connect(**imdb_config)
        cursor = cnx.cursor()
        imdb_table_list = []
        download_data_dir = os.path.join(os.path.dirname(__file__), "imdb_data")
        imdb_dict = {}
        cursor.execute("show tables")
        for table in cursor:
            imdb_table_list.append(table[0])
        for table in imdb_table_list:
            header_list = []
            cursor.execute("desc {}".format(table))
            for column in cursor:
                header_list.append(column[0])
            csv_name = "{}.csv".format(table)
            data_file_name = os.path.join(download_data_dir, csv_name)
            if not os.path.exists(data_file_name):
                if not os.path.exists(download_data_dir):
                    os.makedirs(download_data_dir)
                cursor.execute("select * from {}".format(table))
                result=cursor.fetchall()
                c = csv.writer(open(data_file_name, 'w', encoding='utf-8'))
                c.writerow(header_list)
                for row in result:
                    c.writerow(row)
                print("Created:", csv_name)
            imdb_dict[csv_name.split('.')[0]] = get_data_from_csv(datatype, data_file_name)
        return imdb_dict
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            raise ValueError("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            raise ValueError("Database does not exist")
        else:
            raise ValueError(err)
    else:
        cnx.close()

# d1 = fetch_imdb_dataset('spark')
d1 = fetch_go_sales_dataset('pandas')
print(type(d1['go_1k']))
print(d1['go_1k'])
