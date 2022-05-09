# Copyright 2019-2022 IBM Corporation
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
import urllib.request
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
import sklearn
from packaging import version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sklearn_version = version.parse(getattr(sklearn, "__version__"))

try:
    import arff
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """Package 'arff' not found. You can install it with
    pip install 'liac-arff>=2.4.0'
or with
    pip install 'lale[full]'"""
    )

download_data_dir = os.path.join(os.path.dirname(__file__), "download_data")
experiments_dict: Dict[str, Dict[str, Union[str, int]]] = {}

# 1.25
experiments_dict["vehicle"] = {
    "download_arff_url": "https://www.openml.org/data/download/54/dataset_54_vehicle.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/54/dataset_54_vehicle.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 846,
}

# 1.3
experiments_dict["blood-transfusion-service-center"] = {
    "download_arff_url": "https://www.openml.org/data/download/1586225/php0iVrYT",
    "download_csv_url": "https://www.openml.org/data/get_csv/1586225/php0iVrYT",
    "task_type": "classification",
    "target": "class",
    "n_rows": 748,
}

# 1.5
experiments_dict["car"] = {
    "download_arff_url": "https://www.openml.org/data/download/18116966/php2jDIhh",
    "download_csv_url": "https://www.openml.org/data/get_csv/18116966/php2jDIhh",
    "task_type": "classification",
    "target": "class",
    "n_rows": 1728,
}

# 1.6
experiments_dict["kc1"] = {
    "download_arff_url": "https://www.openml.org/data/download/53950/kc1.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/53950/kc1.arff",
    "task_type": "classification",
    "target": "defects",
    "n_rows": 2109,
}

# 2.6
experiments_dict["Australian"] = {
    "download_arff_url": "https://www.openml.org/data/download/18151910/phpelnJ6y",
    "download_csv_url": "https://www.openml.org/data/get_csv/18151910/phpelnJ6y",
    "task_type": "classification",
    "target": "a15",
    "n_rows": 690,
}

# 3.1
experiments_dict["credit-g"] = {
    "download_arff_url": "https://www.openml.org/data/download/31/dataset_31_credit-g.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 1000,
}

# 3.4
experiments_dict["phoneme"] = {
    "download_arff_url": "https://www.openml.org/data/download/1592281/php8Mz7BG",
    "download_csv_url": "https://www.openml.org/data/get_csv/1592281/php8Mz7BG",
    "task_type": "classification",
    "target": "class",
    "n_rows": 5404,
}

# 3.6
experiments_dict["kr-vs-kp"] = {
    "download_arff_url": "https://www.openml.org/data/download/3/dataset_3_kr-vs-kp.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/3/dataset_3_kr-vs-kp.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 3196,
}

# 4.0
experiments_dict["mfeat-factors"] = {
    "download_arff_url": "https://www.openml.org/data/download/12/dataset_12_mfeat-factors.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/12/dataset_12_mfeat-factors.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 2000,
}

# 5.9
experiments_dict["cnae-9"] = {
    "download_arff_url": "https://www.openml.org/data/download/1586233/phpmcGu2X",
    "download_csv_url": "https://www.openml.org/data/get_csv/1586233/phpmcGu2X",
    "task_type": "classification",
    "target": "class",
    "n_rows": 1080,
}

# 8.1
experiments_dict["sylvine"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335519/file7a97574fa9ae.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335519/file7a97574fa9ae.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 5124,
}

# 17
experiments_dict["jungle_chess_2pcs_raw_endgame_complete"] = {
    "download_arff_url": "https://www.openml.org/data/download/18631418/jungle_chess_2pcs_raw_endgame_complete.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/18631418/jungle_chess_2pcs_raw_endgame_complete.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 44819,
}

# 32
experiments_dict["shuttle"] = {
    "download_arff_url": "https://www.openml.org/data/download/4965262/shuttle.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/4965262/shuttle.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 58000,
}

# 55
experiments_dict["jasmine"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335516/file79b563a1a18.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335516/file79b563a1a18.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 2984,
}

# 118
experiments_dict["fabert"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335687/file1c555f4ca44d.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335687/file1c555f4ca44d.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 8237,
}

# 226
experiments_dict["helena"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335692/file1c556677f875.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335692/file1c556677f875.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 65196,
}

# 230
experiments_dict["bank-marketing"] = {
    "download_arff_url": "https://www.openml.org/data/download/1586218/phpkIxskf",
    "download_csv_url": "https://www.openml.org/data/get_csv/1586218/phpkIxskf",
    "task_type": "classification",
    "target": "class",
    "n_rows": 4521,
}

# 407
experiments_dict["nomao"] = {
    "download_arff_url": "https://www.openml.org/data/download/1592278/phpDYCOet",
    "download_csv_url": "https://www.openml.org/data/get_csv/1592278/phpDYCOet",
    "task_type": "classification",
    "target": "class",
    "n_rows": 34465,
}

# 425
experiments_dict["dilbert"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335686/file1c5552c0c4b0.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335686/file1c5552c0c4b0.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 10000,
}

# 442
experiments_dict["numerai28.6"] = {
    "download_arff_url": "https://www.openml.org/data/download/2160285/phpg2t68G",
    "download_csv_url": "https://www.openml.org/data/get_csv/2160285/phpg2t68G",
    "task_type": "classification",
    "target": "attribute_21",
    "n_rows": 96320,
}

# 457
experiments_dict["prnn_cushings"] = {
    "task_type": "classification",
    "target": "type",
    "download_arff_url": "https://www.openml.org/data/download/52569/prnn_cushings.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/52569/prnn_cushings.csv",
    "n_rows": 27,
}

# 503
experiments_dict["adult"] = {
    "download_arff_url": "https://www.openml.org/data/download/1595261/phpMawTba",
    "download_csv_url": "https://www.openml.org/data/get_csv/1595261/phpMawTba",
    "task_type": "classification",
    "target": "class",
    "n_rows": 48842,
}

# 633
experiments_dict["higgs"] = {
    "download_arff_url": "https://www.openml.org/data/download/2063675/phpZLgL9q",
    "download_csv_url": "https://www.openml.org/data/get_csv/2063675/phpZLgL9q",
    "task_type": "classification",
    "target": "class",
    "n_rows": 98050,
}

# 981
experiments_dict["christine"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335515/file764d5d063390.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335515/file764d5d063390.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 5418,
}

# 1169
experiments_dict["jannis"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335691/file1c558ee247d.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335691/file1c558ee247d.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 83733,
}

# 1503
experiments_dict["connect-4"] = {
    "download_arff_url": "https://www.openml.org/data/download/4965243/connect-4.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/4965243/connect-4.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 67557,
}

# 1580
experiments_dict["volkert"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335689/file1c556e3db171.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335689/file1c556e3db171.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 58310,
}

# 2112
experiments_dict["APSFailure"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335511/aps_failure.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335511/aps_failure.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 76000,
}

# 3700
experiments_dict["riccardo"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335534/file7b535210a7df.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335534/file7b535210a7df.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 20000,
}

# 3759
experiments_dict["guillermo"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335532/file7b5323e77330.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335532/file7b5323e77330.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 20000,
}

experiments_dict["albert"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335520/file7b53746cbda2.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335520/file7b53746cbda2.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 425240,
}

experiments_dict["robert"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335688/file1c55384ec217.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335688/file1c55384ec217.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 10000,
}

experiments_dict["covertype"] = {
    "download_arff_url": "https://www.openml.org/data/download/1601911/phpQOf0wY",
    "download_csv_url": "https://www.openml.org/data/get_csv/1601911/phpQOf0wY",
    "task_type": "classification",
    "target": "class",
    "n_rows": 581012,
}

# This dataset doesn't work with the pre-processing pipeline coded below, as the SimpleImputer drops some columns
# which have all missing values. There is no easy way to pass this info to the downstream ColumnTransformer.
# experiments_dict['KDDCup09_appetency'] = {}
#     'download_arff_url'] = 'https://www.openml.org/data/download/53994/KDDCup09_appetency.arff'
#     'download_csv_url'] = 'https://www.openml.org/data/get_csv/53994/KDDCup09_appetency.arff'
#     'task_type'] = 'classification'
#     'target'] = 'appetency'

experiments_dict["Amazon_employee_access"] = {
    "download_arff_url": "https://www.openml.org/data/download/1681098/phpmPOD5A",
    "download_csv_url": "https://www.openml.org/data/get_csv/1681098/phpmPOD5A",
    "task_type": "classification",
    "target": "target",
    "n_rows": 32769,
}

experiments_dict["Fashion-MNIST"] = {
    "download_arff_url": "https://www.openml.org/data/download/18238735/phpnBqZGZ",
    "download_csv_url": "https://www.openml.org/data/get_csv/18238735/phpnBqZGZ",
    "task_type": "classification",
    "target": "class",
    "n_rows": 70000,
}

experiments_dict["dionis"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335690/file1c55272d7b5b.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335690/file1c55272d7b5b.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 416188,
}

experiments_dict["MiniBooNE"] = {
    "download_arff_url": "https://www.openml.org/data/download/19335523/MiniBooNE.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/19335523/MiniBooNE.arff",
    "task_type": "classification",
    "target": "signal",
    "n_rows": 130064,
}

experiments_dict["airlines"] = {
    "download_arff_url": "https://www.openml.org/data/download/66526/phpvcoG8S",
    "download_csv_url": "https://www.openml.org/data/get_csv/66526/phpvcoG8S",
    "task_type": "stream classification",
    "target": "class",
    "n_rows": 539383,
}

experiments_dict["diabetes"] = {
    "dataset_url": "https://www.openml.org/d/37",
    "download_arff_url": "https://www.openml.org/data/download/37/dataset_37_diabetes.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/37/dataset_37_diabetes.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 768,
}

experiments_dict["spectf"] = {
    "dataset_url": "https://www.openml.org/d/337",
    "download_arff_url": "https://www.openml.org/data/download/52240/phpDQbeeh",
    "download_csv_url": "https://www.openml.org/data/get_csv/52240/phpDQbeeh",
    "task_type": "classification",
    "target": "overall_diagnosis",
    "n_rows": 267,
}

experiments_dict["hill-valley"] = {
    "dataset_url": "https://www.openml.org/d/1479",
    "download_arff_url": "https://www.openml.org/data/download/1590101/php3isjYz",
    "download_csv_url": "https://www.openml.org/data/get_csv/1590101/php3isjYz",
    "task_type": "classification",
    "target": "class",
    "n_rows": 1212,
}

experiments_dict["breast-cancer"] = {
    "dataset_url": "https://www.openml.org/d/13",
    "download_arff_url": "https://www.openml.org/data/download/13/dataset_13_breast-cancer.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/13/dataset_13_breast-cancer.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 286,
}

experiments_dict["compas"] = {
    "download_arff_url": "https://www.openml.org/data/download/21757035/compas.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/21757035/compas.arff",
    "task_type": "classification",
    "target": "two_year_recid",
    "n_rows": 5278,
}

experiments_dict["ricci"] = {
    "download_arff_url": "https://www.openml.org/data/download/22044446/ricci_processed.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/22044446/ricci_processed.arff",
    "task_type": "classification",
    "target": "promotion",
    "n_rows": 118,
}

experiments_dict["SpeedDating"] = {
    "dataset_url": "https://www.openml.org/d/40536",
    "download_arff_url": "https://www.openml.org/data/download/13153954/speeddating.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/13153954/speeddating.arff",
    "task_type": "classification",
    "target": "match",
    "n_rows": 8378,
}

experiments_dict["nursery"] = {
    "dataset_url": "https://www.openml.org/d/26",
    "download_arff_url": "https://www.openml.org/data/download/26/dataset_26_nursery.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/26/dataset_26_nursery.arff",
    "task_type": "classification",
    "target": "class",
    "n_rows": 12960,
}

experiments_dict["titanic"] = {
    "dataset_url": "https://www.openml.org/d/40945",
    "download_arff_url": "https://www.openml.org/data/download/16826755/phpMYEkMl",
    "download_csv_url": "https://www.openml.org/data/get_csv/16826755/phpMYEkMl",
    "task_type": "classification",
    "target": "survived",
    "n_rows": 1309,
}

experiments_dict["tae"] = {
    "dataset_url": "https://www.openml.org/d/48",
    "download_arff_url": "https://www.openml.org/data/download/48/dataset_48_tae.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/48/dataset_48_tae.arff",
    "task_type": "classification",
    "target": "class_attribute",
    "n_rows": 151,
}

experiments_dict["airlines_delay"] = {
    "dataset_url": "https://www.openml.org/d/42728",
    "download_arff_url": "https://www.openml.org/data/download/22044760/airlines_train_regression_10000000.arff",
    "download_csv_url": "https://www.openml.org/data/get_csv/22044760/airlines_train_regression_10000000.arff",
    "task_type": "regression",
    "target": "depdelay",
    "n_rows": 10000000,
}

experiments_dict["kddcup99full"] = {
    "dataset_url": "https://www.openml.org/d/42728",
    "download_arff_url": "https://www.openml.org/data/download/53993/KDDCup99_full.arff",
    "download_csv_url": "https://www.openml.org/data/download/53993/KDDCup99_full.arff",
    "task_type": "classification",
    "target": "label",
    "n_rows": 4898431,
}


def add_schemas(schema_orig, target_col, train_X, test_X, train_y, test_y):
    from lale.datasets.data_schemas import add_schema

    elems_X = [
        item_schema
        for item_schema in schema_orig["items"]["items"]
        if item_schema["description"].lower() != target_col
    ]
    elem_y = [
        item_schema
        for item_schema in schema_orig["items"]["items"]
        if item_schema["description"].lower() == target_col
    ][0]
    if "enum" in elem_y:
        if isinstance(train_y, pd.Series):
            elem_y["enum"] = list(train_y.unique())
        else:
            elem_y["enum"] = [*range(len(elem_y["enum"]))]
    ncols_X = len(elems_X)
    rows_X = {
        **schema_orig["items"],
        "minItems": ncols_X,
        "maxItems": ncols_X,
        "items": elems_X,
    }
    if "json_schema" not in pd.DataFrame._internal_names:
        pd.DataFrame._internal_names.append("json_schema")
    nrows_train, nrows_test = len(train_y), len(test_y)
    train_X = add_schema(
        train_X,
        {
            **schema_orig,
            "minItems": nrows_train,
            "maxItems": nrows_train,
            "items": rows_X,
        },
    )
    test_X = add_schema(
        test_X,
        {
            **schema_orig,
            "minItems": nrows_test,
            "maxItems": nrows_test,
            "items": rows_X,
        },
    )
    train_y = add_schema(
        train_y,
        {
            **schema_orig,
            "minItems": nrows_train,
            "maxItems": nrows_train,
            "items": elem_y,
        },
    )
    test_y = add_schema(
        test_y,
        {
            **schema_orig,
            "minItems": nrows_test,
            "maxItems": nrows_test,
            "items": elem_y,
        },
    )
    return train_X, test_X, train_y, test_y


numeric_data_types_list = ["numeric", "integer", "real"]


def download_if_missing(dataset_name, verbose=False):
    file_name = os.path.join(download_data_dir, dataset_name + ".arff")
    is_missing = not os.path.exists(file_name)
    if verbose:
        print(
            f"download_if_missing('{dataset_name}'): is_missing {is_missing}, file_name '{file_name}'"
        )
    if is_missing:
        if not os.path.exists(download_data_dir):
            os.makedirs(download_data_dir)
        url = cast(str, experiments_dict[dataset_name]["download_arff_url"])
        urllib.request.urlretrieve(url, file_name)
    assert os.path.exists(file_name)
    return file_name


def fetch(
    dataset_name,
    task_type,
    verbose=False,
    preprocess=True,
    test_size=0.33,
    astype=None,
    seed=0,
):
    # Check that the dataset name exists in experiments_dict
    try:
        if experiments_dict[dataset_name]["task_type"] != task_type.lower():
            raise ValueError(
                "The task type {} does not match with the given datasets task type {}".format(
                    task_type, experiments_dict[dataset_name]["task_type"]
                )
            )
    except KeyError:
        raise KeyError(
            "Dataset name {} not found in the supported datasets".format(dataset_name)
        )

    data_file_name = download_if_missing(dataset_name, verbose)
    with open(data_file_name) as f:
        dataDictionary = arff.load(f)
        f.close()

    from lale.datasets.data_schemas import liac_arff_to_schema

    schema_orig = liac_arff_to_schema(dataDictionary)
    target_col = experiments_dict[dataset_name]["target"]
    y: Optional[Any] = None
    if preprocess:
        arffData = pd.DataFrame(dataDictionary["data"])
        # arffData = arffData.fillna(0)
        attributes = dataDictionary["attributes"]

        if verbose:
            print(f"attributes: {attributes}")
        categorical_cols = []
        numeric_cols = []
        X_columns = []
        for i, item in enumerate(attributes):
            if item[0].lower() == target_col:
                target_indx = i
                # remove it from attributes so that the next loop indices are adjusted accordingly.
                del attributes[i]
                # the type stubs for pandas are not currently complete enough to type this correctly
                y = arffData.iloc[:, target_indx]  # type: ignore
                arffData = arffData.drop(i, axis=1)

        for i, item in enumerate(attributes):
            X_columns.append(i)
            if (
                (
                    isinstance(item[1], str)
                    and item[1].lower() not in numeric_data_types_list
                )
                or isinstance(item[1], list)
            ) and (item[0].lower() != "class"):
                categorical_cols.append(i)
            elif (
                isinstance(item[1], str) and item[1].lower() in numeric_data_types_list
            ) and (item[0].lower() != "class"):
                numeric_cols.append(i)
        if verbose:
            print(f"categorical columns: {categorical_cols}")
            print(f"numeric columns:     {numeric_cols}")
        X = arffData.iloc[:, X_columns]

        # Check whether there is any error
        num_classes_from_last_row = len(list(set(y))) if y is not None else 0

        if verbose:
            print("num_classes_from_last_row", num_classes_from_last_row)

        transformers1 = [
            (
                "imputer_str",
                SimpleImputer(missing_values=None, strategy="most_frequent"),
                categorical_cols,
            ),
            ("imputer_num", SimpleImputer(strategy="mean"), numeric_cols),
        ]
        txm1 = ColumnTransformer(transformers1, sparse_threshold=0.0)

        transformers2 = [
            ("ohe", OneHotEncoder(sparse=False), list(range(len(categorical_cols)))),
            (
                "no_op",
                "passthrough",
                list(
                    range(
                        len(categorical_cols), len(categorical_cols) + len(numeric_cols)
                    )
                ),
            ),
        ]
        txm2 = ColumnTransformer(transformers2, sparse_threshold=0.0)
        if verbose:
            print("Shape of X before preprocessing", X.shape)
        from sklearn.pipeline import make_pipeline

        preprocessing = make_pipeline(txm1, txm2)

        X = preprocessing.fit(X).transform(X)

        if verbose:
            print(f"shape of X after preprocessing: {X.shape}")

        if astype in ["pandas", "spark", "spark-with-index"]:
            cat_col_names = [attributes[i][0].lower() for i in categorical_cols]
            one_hot_encoder = preprocessing.steps[1][1].named_transformers_["ohe"]
            if sklearn_version >= version.Version("1.0"):
                encoded_names = one_hot_encoder.get_feature_names_out(cat_col_names)
            else:
                encoded_names = one_hot_encoder.get_feature_names(cat_col_names)
            num_col_names = [attributes[i][0].lower() for i in numeric_cols]
            col_names = list(encoded_names) + list(num_col_names)
            if verbose:
                print(f"column names after preprocessing: {col_names}")
            X = pd.DataFrame(X, columns=col_names)

    else:
        col_names = [attr[0].lower() for attr in dataDictionary["attributes"]]
        df_all = pd.DataFrame(dataDictionary["data"], columns=col_names)
        y = df_all[target_col]
        # the type stubs for pandas are not currently complete enough to type this correctly
        y = y.squeeze()  # type: ignore
        cols_X = [col for col in col_names if col != target_col]
        X = df_all[cols_X]

    if preprocess:
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
    if astype in ["pandas", "spark", "spark-with-index"] and not isinstance(
        y, pd.Series
    ):
        y = pd.Series(y, name=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    if verbose:
        print(f"training set shapes: X {X_train.shape}, y {y_train.shape}")
        print(f"test set shapes:     X {X_test.shape}, y {y_test.shape}")
    if preprocess:
        from lale.datasets.data_schemas import add_schema

        X_train = add_schema(X_train.astype(np.number), recalc=True)
        y_train = add_schema(y_train.astype(int), recalc=True)
        X_test = add_schema(X_test.astype(np.number), recalc=True)
        y_test = add_schema(y_test.astype(int), recalc=True)
    else:
        X_train, X_test, y_train, y_test = add_schemas(
            schema_orig, target_col, X_train, X_test, y_train, y_test
        )
    if astype == "spark":
        from lale.datasets import pandas2spark

        X_train = pandas2spark(X_train)
        X_test = pandas2spark(X_test)
    if astype == "spark-with-index":
        from lale.datasets import pandas2spark

        X_train = pandas2spark(X_train, with_index=True)
        X_test = pandas2spark(X_test, with_index=True)
    return (X_train, y_train), (X_test, y_test)
