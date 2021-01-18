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
import urllib.request
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
experiments_dict: Dict[str, Dict[str, str]] = {}

# 1.25
experiments_dict["vehicle"] = {}
experiments_dict["vehicle"][
    "download_arff_url"
] = "https://www.openml.org/data/download/54/dataset_54_vehicle.arff"
experiments_dict["vehicle"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/54/dataset_54_vehicle.arff"
experiments_dict["vehicle"]["task_type"] = "classification"
experiments_dict["vehicle"]["target"] = "class"

# 1.3
experiments_dict["blood-transfusion-service-center"] = {}
experiments_dict["blood-transfusion-service-center"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1586225/php0iVrYT"
experiments_dict["blood-transfusion-service-center"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1586225/php0iVrYT"
experiments_dict["blood-transfusion-service-center"]["task_type"] = "classification"
experiments_dict["blood-transfusion-service-center"]["target"] = "class"

# 1.5
experiments_dict["car"] = {}
experiments_dict["car"][
    "download_arff_url"
] = "https://www.openml.org/data/download/18116966/php2jDIhh"
experiments_dict["car"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/18116966/php2jDIhh"
experiments_dict["car"]["task_type"] = "classification"
experiments_dict["car"]["target"] = "class"

# 1.6
experiments_dict["kc1"] = {}
experiments_dict["kc1"][
    "download_arff_url"
] = "https://www.openml.org/data/download/53950/kc1.arff"
experiments_dict["kc1"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/53950/kc1.arff"
experiments_dict["kc1"]["task_type"] = "classification"
experiments_dict["kc1"]["target"] = "defects"

# 2.6
experiments_dict["Australian"] = {}
experiments_dict["Australian"][
    "download_arff_url"
] = "https://www.openml.org/data/download/18151910/phpelnJ6y"
experiments_dict["Australian"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/18151910/phpelnJ6y"
experiments_dict["Australian"]["task_type"] = "classification"
experiments_dict["Australian"]["target"] = "a15"

# 3.1
experiments_dict["credit-g"] = {}
experiments_dict["credit-g"][
    "download_arff_url"
] = "https://www.openml.org/data/download/31/dataset_31_credit-g.arff"
experiments_dict["credit-g"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff"
experiments_dict["credit-g"]["task_type"] = "classification"
experiments_dict["credit-g"]["target"] = "class"

# 3.4
experiments_dict["phoneme"] = {}
experiments_dict["phoneme"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1592281/php8Mz7BG"
experiments_dict["phoneme"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1592281/php8Mz7BG"
experiments_dict["phoneme"]["task_type"] = "classification"
experiments_dict["phoneme"]["target"] = "class"

# 3.6
experiments_dict["kr-vs-kp"] = {}
experiments_dict["kr-vs-kp"][
    "download_arff_url"
] = "https://www.openml.org/data/download/3/dataset_3_kr-vs-kp.arff"
experiments_dict["kr-vs-kp"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/3/dataset_3_kr-vs-kp.arff"
experiments_dict["kr-vs-kp"]["task_type"] = "classification"
experiments_dict["kr-vs-kp"]["target"] = "class"

# 4.0
experiments_dict["mfeat-factors"] = {}
experiments_dict["mfeat-factors"][
    "download_arff_url"
] = "https://www.openml.org/data/download/12/dataset_12_mfeat-factors.arff"
experiments_dict["mfeat-factors"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/12/dataset_12_mfeat-factors.arff"
experiments_dict["mfeat-factors"]["task_type"] = "classification"
experiments_dict["mfeat-factors"]["target"] = "class"

# 5.9
experiments_dict["cnae-9"] = {}
experiments_dict["cnae-9"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1586233/phpmcGu2X"
experiments_dict["cnae-9"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1586233/phpmcGu2X"
experiments_dict["cnae-9"]["task_type"] = "classification"
experiments_dict["cnae-9"]["target"] = "class"

# 8.1
experiments_dict["sylvine"] = {}
experiments_dict["sylvine"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335519/file7a97574fa9ae.arff"
experiments_dict["sylvine"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335519/file7a97574fa9ae.arff"
experiments_dict["sylvine"]["task_type"] = "classification"
experiments_dict["sylvine"]["target"] = "class"

# 17
experiments_dict["jungle_chess_2pcs_raw_endgame_complete"] = {}
experiments_dict["jungle_chess_2pcs_raw_endgame_complete"][
    "download_arff_url"
] = "https://www.openml.org/data/download/18631418/jungle_chess_2pcs_raw_endgame_complete.arff"
experiments_dict["jungle_chess_2pcs_raw_endgame_complete"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/18631418/jungle_chess_2pcs_raw_endgame_complete.arff"
experiments_dict["jungle_chess_2pcs_raw_endgame_complete"][
    "task_type"
] = "classification"
experiments_dict["jungle_chess_2pcs_raw_endgame_complete"]["target"] = "class"

# 32
experiments_dict["shuttle"] = {}
experiments_dict["shuttle"][
    "download_arff_url"
] = "https://www.openml.org/data/download/4965262/shuttle.arff"
experiments_dict["shuttle"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/4965262/shuttle.arff"
experiments_dict["shuttle"]["task_type"] = "classification"
experiments_dict["shuttle"]["target"] = "class"

# 55
experiments_dict["jasmine"] = {}
experiments_dict["jasmine"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335516/file79b563a1a18.arff"
experiments_dict["jasmine"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335516/file79b563a1a18.arff"
experiments_dict["jasmine"]["task_type"] = "classification"
experiments_dict["jasmine"]["target"] = "class"

# 118
experiments_dict["fabert"] = {}
experiments_dict["fabert"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335687/file1c555f4ca44d.arff"
experiments_dict["fabert"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335687/file1c555f4ca44d.arff"
experiments_dict["fabert"]["task_type"] = "classification"
experiments_dict["fabert"]["target"] = "class"

# 226
experiments_dict["helena"] = {}
experiments_dict["helena"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335692/file1c556677f875.arff"
experiments_dict["helena"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335692/file1c556677f875.arff"
experiments_dict["helena"]["task_type"] = "classification"
experiments_dict["helena"]["target"] = "class"

# 230
experiments_dict["bank-marketing"] = {}
experiments_dict["bank-marketing"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1586218/phpkIxskf"
experiments_dict["bank-marketing"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1586218/phpkIxskf"
experiments_dict["bank-marketing"]["task_type"] = "classification"
experiments_dict["bank-marketing"]["target"] = "class"

# 407
experiments_dict["nomao"] = {}
experiments_dict["nomao"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1592278/phpDYCOet"
experiments_dict["nomao"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1592278/phpDYCOet"
experiments_dict["nomao"]["task_type"] = "classification"
experiments_dict["nomao"]["target"] = "class"

# 425
experiments_dict["dilbert"] = {}
experiments_dict["dilbert"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335686/file1c5552c0c4b0.arff"
experiments_dict["dilbert"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335686/file1c5552c0c4b0.arff"
experiments_dict["dilbert"]["task_type"] = "classification"
experiments_dict["dilbert"]["target"] = "class"

# 442
experiments_dict["numerai28.6"] = {}
experiments_dict["numerai28.6"][
    "download_arff_url"
] = "https://www.openml.org/data/download/2160285/phpg2t68G"
experiments_dict["numerai28.6"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/2160285/phpg2t68G"
experiments_dict["numerai28.6"]["task_type"] = "classification"
experiments_dict["numerai28.6"]["target"] = "attribute_21"

# 503
experiments_dict["adult"] = {}
experiments_dict["adult"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1595261/phpMawTba"
experiments_dict["adult"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1595261/phpMawTba"
experiments_dict["adult"]["task_type"] = "classification"
experiments_dict["adult"]["target"] = "class"

# 633
experiments_dict["higgs"] = {}
experiments_dict["higgs"][
    "download_arff_url"
] = "https://www.openml.org/data/download/2063675/phpZLgL9q"
experiments_dict["higgs"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/2063675/phpZLgL9q"
experiments_dict["higgs"]["task_type"] = "classification"
experiments_dict["higgs"]["target"] = "class"

# 981
experiments_dict["christine"] = {}
experiments_dict["christine"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335515/file764d5d063390.arff"
experiments_dict["christine"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335515/file764d5d063390.arff"
experiments_dict["christine"]["task_type"] = "classification"
experiments_dict["christine"]["target"] = "class"

# 1169
experiments_dict["jannis"] = {}
experiments_dict["jannis"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335691/file1c558ee247d.arff"
experiments_dict["jannis"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335691/file1c558ee247d.arff"
experiments_dict["jannis"]["task_type"] = "classification"
experiments_dict["jannis"]["target"] = "class"

# 1503
experiments_dict["connect-4"] = {}
experiments_dict["connect-4"][
    "download_arff_url"
] = "https://www.openml.org/data/download/4965243/connect-4.arff"
experiments_dict["connect-4"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/4965243/connect-4.arff"
experiments_dict["connect-4"]["task_type"] = "classification"
experiments_dict["connect-4"]["target"] = "class"

# 1580
experiments_dict["volkert"] = {}
experiments_dict["volkert"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335689/file1c556e3db171.arff"
experiments_dict["volkert"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335689/file1c556e3db171.arff"
experiments_dict["volkert"]["task_type"] = "classification"
experiments_dict["volkert"]["target"] = "class"

# 2112
experiments_dict["APSFailure"] = {}
experiments_dict["APSFailure"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335511/aps_failure.arff"
experiments_dict["APSFailure"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335511/aps_failure.arff"
experiments_dict["APSFailure"]["task_type"] = "classification"
experiments_dict["APSFailure"]["target"] = "class"

# 3700
experiments_dict["riccardo"] = {}
experiments_dict["riccardo"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335534/file7b535210a7df.arff"
experiments_dict["riccardo"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335534/file7b535210a7df.arff"
experiments_dict["riccardo"]["task_type"] = "classification"
experiments_dict["riccardo"]["target"] = "class"

# 3759
experiments_dict["guillermo"] = {}
experiments_dict["guillermo"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335532/file7b5323e77330.arff"
experiments_dict["guillermo"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335532/file7b5323e77330.arff"
experiments_dict["guillermo"]["task_type"] = "classification"
experiments_dict["guillermo"]["target"] = "class"

experiments_dict["albert"] = {}
experiments_dict["albert"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335520/file7b53746cbda2.arff"
experiments_dict["albert"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335520/file7b53746cbda2.arff"
experiments_dict["albert"]["task_type"] = "classification"
experiments_dict["albert"]["target"] = "class"

experiments_dict["robert"] = {}
experiments_dict["robert"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335688/file1c55384ec217.arff"
experiments_dict["robert"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335688/file1c55384ec217.arff"
experiments_dict["robert"]["task_type"] = "classification"
experiments_dict["robert"]["target"] = "class"

experiments_dict["covertype"] = {}
experiments_dict["covertype"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1601911/phpQOf0wY"
experiments_dict["covertype"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1601911/phpQOf0wY"
experiments_dict["covertype"]["task_type"] = "classification"
experiments_dict["covertype"]["target"] = "class"

# This dataset doesn't work with the pre-processing pipeline coded below, as the SimpleImputer drops some columns
# which have all missing values. There is no easy way to pass this info to the downstream ColumnTransformer.
# experiments_dict['KDDCup09_appetency'] = {}
# experiments_dict['KDDCup09_appetency']['download_arff_url'] = 'https://www.openml.org/data/download/53994/KDDCup09_appetency.arff'
# experiments_dict['KDDCup09_appetency']['download_csv_url'] = 'https://www.openml.org/data/get_csv/53994/KDDCup09_appetency.arff'
# experiments_dict['KDDCup09_appetency']['task_type'] = 'classification'
# experiments_dict['KDDCup09_appetency']['target'] = 'appetency'

experiments_dict["Amazon_employee_access"] = {}
experiments_dict["Amazon_employee_access"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1681098/phpmPOD5A"
experiments_dict["Amazon_employee_access"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1681098/phpmPOD5A"
experiments_dict["Amazon_employee_access"]["task_type"] = "classification"
experiments_dict["Amazon_employee_access"]["target"] = "target"

experiments_dict["Fashion-MNIST"] = {}
experiments_dict["Fashion-MNIST"][
    "download_arff_url"
] = "https://www.openml.org/data/download/18238735/phpnBqZGZ"
experiments_dict["Fashion-MNIST"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/18238735/phpnBqZGZ"
experiments_dict["Fashion-MNIST"]["task_type"] = "classification"
experiments_dict["Fashion-MNIST"]["target"] = "class"

experiments_dict["dionis"] = {}
experiments_dict["dionis"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335690/file1c55272d7b5b.arff"
experiments_dict["dionis"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335690/file1c55272d7b5b.arff"
experiments_dict["dionis"]["task_type"] = "classification"
experiments_dict["dionis"]["target"] = "class"

experiments_dict["MiniBooNE"] = {}
experiments_dict["MiniBooNE"][
    "download_arff_url"
] = "https://www.openml.org/data/download/19335523/MiniBooNE.arff"
experiments_dict["MiniBooNE"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/19335523/MiniBooNE.arff"
experiments_dict["MiniBooNE"]["task_type"] = "classification"
experiments_dict["MiniBooNE"]["target"] = "signal"

experiments_dict["airlines"] = {}
experiments_dict["airlines"][
    "download_arff_url"
] = "https://www.openml.org/data/download/66526/phpvcoG8S"
experiments_dict["airlines"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/66526/phpvcoG8S"
experiments_dict["airlines"]["task_type"] = "stream classification"
experiments_dict["airlines"]["target"] = "class"

experiments_dict["diabetes"] = {}
experiments_dict["diabetes"]["dataset_url"] = "https://www.openml.org/d/37"
experiments_dict["diabetes"][
    "download_arff_url"
] = "https://www.openml.org/data/download/37/dataset_37_diabetes.arff"
experiments_dict["diabetes"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/37/dataset_37_diabetes.arff"
experiments_dict["diabetes"]["task_type"] = "classification"
experiments_dict["diabetes"]["target"] = "class"

experiments_dict["spectf"] = {}
experiments_dict["spectf"]["dataset_url"] = "https://www.openml.org/d/337"
experiments_dict["spectf"][
    "download_arff_url"
] = "https://www.openml.org/data/download/52240/phpDQbeeh"
experiments_dict["spectf"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/52240/phpDQbeeh"
experiments_dict["spectf"]["task_type"] = "classification"
experiments_dict["spectf"]["target"] = "overall_diagnosis"

experiments_dict["hill-valley"] = {}
experiments_dict["hill-valley"]["dataset_url"] = "https://www.openml.org/d/1479"
experiments_dict["hill-valley"][
    "download_arff_url"
] = "https://www.openml.org/data/download/1590101/php3isjYz"
experiments_dict["hill-valley"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/1590101/php3isjYz"
experiments_dict["hill-valley"]["task_type"] = "classification"
experiments_dict["hill-valley"]["target"] = "class"

experiments_dict["breast-cancer"] = {}
experiments_dict["breast-cancer"]["dataset_url"] = "https://www.openml.org/d/13"
experiments_dict["breast-cancer"][
    "download_arff_url"
] = "https://www.openml.org/data/download/13/dataset_13_breast-cancer.arff"
experiments_dict["breast-cancer"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/13/dataset_13_breast-cancer.arff"
experiments_dict["breast-cancer"]["task_type"] = "classification"
experiments_dict["breast-cancer"]["target"] = "class"

experiments_dict["compas"] = {}
experiments_dict["compas"][
    "download_arff_url"
] = "https://www.openml.org/data/download/21757035/compas.arff"
experiments_dict["compas"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/21757035/compas.arff"
experiments_dict["compas"]["task_type"] = "classification"
experiments_dict["compas"]["target"] = "two_year_recid"

experiments_dict["ricci"] = {}
experiments_dict["ricci"][
    "download_arff_url"
] = "https://www.openml.org/data/download/22044446/ricci_processed.arff"
experiments_dict["ricci"][
    "download_csv_url"
] = "https://www.openml.org/data/get_csv/22044446/ricci_processed.arff"
experiments_dict["ricci"]["task_type"] = "classification"
experiments_dict["ricci"]["target"] = "promotion"


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


def fetch(
    dataset_name, task_type, verbose=False, preprocess=True, test_size=0.33, astype=None
):
    if verbose:
        print("Loading dataset:", dataset_name)
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
    data_file_name = os.path.join(download_data_dir, dataset_name + ".arff")
    if verbose:
        print(f"data file name: {data_file_name}")
    if not os.path.exists(data_file_name):
        # TODO: Download the data
        if not os.path.exists(download_data_dir):
            os.makedirs(download_data_dir)
            if verbose:
                print("created directory {}".format(download_data_dir))
        urllib.request.urlretrieve(
            experiments_dict[dataset_name]["download_arff_url"], data_file_name
        )

    assert os.path.exists(data_file_name)
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

        if astype == "pandas":
            cat_col_names = [attributes[i][0].lower() for i in categorical_cols]
            one_hot_encoder = preprocessing.steps[1][1].named_transformers_["ohe"]
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
    if astype == "pandas" and not isinstance(y, pd.Series):
        y = pd.Series(y, name=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    if verbose:
        print(f"training set shapes: X {X_train.shape}, y {y_train.shape}")
        print(f"test set shapes:     X {X_test.shape}, y {y_test.shape}")
    if preprocess:
        from lale.datasets.data_schemas import add_schema

        X_train = add_schema(X_train.astype(np.number), recalc=True)
        y_train = add_schema(y_train.astype(np.int), recalc=True)
        X_test = add_schema(X_test.astype(np.number), recalc=True)
        y_test = add_schema(y_test.astype(np.int), recalc=True)
    else:
        X_train, X_test, y_train, y_test = add_schemas(
            schema_orig, target_col, X_train, X_test, y_train, y_test
        )
    return (X_train, y_train), (X_test, y_test)
