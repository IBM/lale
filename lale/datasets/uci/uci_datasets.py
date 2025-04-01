# Copyright 2019-2023 IBM Corporation
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
from lale.datasets.util import download_data_cache_dir

download_data_dir = download_data_cache_dir / "uci" / "download_data"
download_data_url = "http://archive.ics.uci.edu/static/public"


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
            # this request is to a string that begins with a hardcoded http url, so does not risk leaking local data
            urllib.request.urlretrieve(zip_url, tmp_zip_file.name)  # nosec
            with zipfile.ZipFile(tmp_zip_file.name) as myzip:
                for full, base in zip(full_file_names, contents_files):
                    if not os.path.exists(full):
                        myzip.extract(base, data_dir)
    assert all_downloaded()
    return full_file_names


def tsv_to_Xy(file_name, target_col, schema_orig, index_col=None):
    data_all = pd.read_csv(file_name, sep="\t", index_col=index_col)
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


def fetch_drugslib():
    files = download(
        "461",
        "drug+review+dataset+druglib+com.zip",
        ["drugLibTest_raw.tsv", "drugLibTrain_raw.tsv"],
    )
    target_col = "rating"
    json_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "items": {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "items": [
                # index: {"description": "reviewID", "type": "integer", "minimum": 0},
                {"description": "urlDrugName", "type": "string"},
                {
                    "description": "rating",
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                {
                    "description": "effectiveness",
                    "anyOf": [{"type": "string"}, {"enum": [np.nan]}],
                },
                {
                    "description": "sideEffects",
                    "anyOf": [{"type": "string"}, {"enum": [np.nan]}],
                },
                {
                    "description": "condition",
                    "anyOf": [{"type": "string"}, {"enum": [np.nan]}],
                },
                {
                    "description": "benefitsReview",
                    "anyOf": [{"type": "string"}, {"enum": [np.nan]}],
                },
                {
                    "description": "sideEffectsReview",
                    "anyOf": [{"type": "string"}, {"enum": [np.nan]}],
                },
                {
                    "description": "commentsReview",
                    "anyOf": [{"type": "string"}, {"enum": [np.nan]}],
                },
            ],
        },
    }
    test_X, test_y = tsv_to_Xy(files[0], target_col, json_schema, index_col=[0])
    train_X, train_y = tsv_to_Xy(files[1], target_col, json_schema, index_col=[0])
    return train_X, train_y, test_X, test_y


def fetch_household_power_consumption():
    file_name = download(
        "235",
        "individual+household+electric+power+consumption.zip",
        ["household_power_consumption.txt"],
    )
    df = pd.read_csv(file_name[0], sep=";")
    return df
