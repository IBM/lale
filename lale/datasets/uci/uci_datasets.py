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

import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import lale.datasets.data_schemas

download_data_dir = os.path.join(os.path.dirname(__file__), "download_data")
download_data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases"


def download(dataset_id, zip_name, contents_files):
    zip_url = f"{download_data_url}/{dataset_id}/{zip_name}"
    data_dir = os.path.join(download_data_dir, dataset_id)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    full_file_names = [os.path.join(data_dir, base) for base in contents_files]

    def all_downloaded():
        for full in full_file_names:
            if not os.path.exists(full):
                return False
        return True

    if not all_downloaded():
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip_file:
            urllib.request.urlretrieve(zip_url, tmp_zip_file.name)
            with zipfile.ZipFile(tmp_zip_file.name) as myzip:
                for i in range(len(contents_files)):
                    full, base = full_file_names[i], contents_files[i]
                    if not os.path.exists(full):
                        myzip.extract(base, data_dir)
    assert all_downloaded
    return full_file_names


def tsv_to_Xy(file_name, target_col, schema_orig):
    data_all = pd.read_csv(file_name, sep="\t")
    row_schema_X = [
        col_schema
        for col_schema in schema_orig["items"]["items"]
        if col_schema["description"] != target_col
    ]
    columns_X = [col_schema["description"] for col_schema in row_schema_X]
    data_X = data_all.loc[:, columns_X]
    nrows, ncols_X = data_X.shape
    schema_X = {
        **schema_orig,
        "minItems": nrows,
        "maxItems": nrows,
        "items": {
            "type": "array",
            "minItems": ncols_X,
            "maxItems": ncols_X,
            "items": row_schema_X,
        },
    }
    data_X = lale.datasets.data_schemas.add_schema(data_X, schema_X)
    row_schema_y = [
        col_schema
        for col_schema in schema_orig["items"]["items"]
        if col_schema["description"] == target_col
    ]
    data_y = data_all[target_col]
    schema_y = {
        **schema_orig,
        "minItems": nrows,
        "maxItems": nrows,
        "items": row_schema_y[0],
    }
    data_y = lale.datasets.data_schemas.add_schema(data_y, schema_y)
    return data_X, data_y


def fetch_drugscom():
    files = download(
        "00462", "drugsCom_raw.zip", ["drugsComTest_raw.tsv", "drugsComTrain_raw.tsv"]
    )
    target_col = "rating"
    json_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "items": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": [
                {"description": "drugName", "type": "string"},
                {
                    "description": "condition",
                    "anyOf": [{"type": "string"}, {"enum": [np.NaN]}],
                },
                {"description": "review", "type": "string"},
                {
                    "description": "rating",
                    "enum": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                },
                {"description": "date", "type": "string"},
                {"description": "usefulCount", "type": "integer", "minimum": 0},
            ],
        },
    }
    test_X, test_y = tsv_to_Xy(files[0], target_col, json_schema)
    train_X, train_y = tsv_to_Xy(files[1], target_col, json_schema)
    return train_X, train_y, test_X, test_y
