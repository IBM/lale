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

import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import lale.datasets.data_schemas


def _bunch_to_df(bunch, schema_X, schema_y, test_size=0.2, random_state=42):
    train_X_arr, test_X_arr, train_y_arr, test_y_arr = train_test_split(
        bunch.data, bunch.target, test_size=test_size, random_state=random_state
    )
    feature_schemas = schema_X["items"]["items"]
    if isinstance(feature_schemas, list):
        feature_names = [f["description"] for f in feature_schemas]
    else:
        feature_names = [f"x{i}" for i in range(schema_X["items"]["maxItems"])]
    train_X_df = pd.DataFrame(train_X_arr, columns=feature_names)
    test_X_df = pd.DataFrame(test_X_arr, columns=feature_names)
    train_y_df = pd.Series(train_y_arr, name="target")
    test_y_df = pd.Series(test_y_arr, name="target")
    train_nrows, test_nrows = train_X_df.shape[0], test_X_df.shape[0]
    train_X = lale.datasets.data_schemas.add_schema(
        train_X_df, {**schema_X, "minItems": train_nrows, "maxItems": train_nrows}
    )
    test_X = lale.datasets.data_schemas.add_schema(
        test_X_df, {**schema_X, "minItems": test_nrows, "maxItems": test_nrows}
    )
    train_y = lale.datasets.data_schemas.add_schema(
        train_y_df, {**schema_y, "minItems": train_nrows, "maxItems": train_nrows}
    )
    test_y = lale.datasets.data_schemas.add_schema(
        test_y_df, {**schema_y, "minItems": test_nrows, "maxItems": test_nrows}
    )
    return (train_X, train_y), (test_X, test_y)


def load_iris_df(test_size=0.2):
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target
    target_name = "target"
    X, y = shuffle(iris.data, iris.target, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_train_df = pd.DataFrame(X_train, columns=iris.feature_names)
    y_train_df = pd.Series(y_train, name=target_name)

    X_test_df = pd.DataFrame(X_test, columns=iris.feature_names)
    y_test_df = pd.Series(y_test, name=target_name)

    return (X_train_df, y_train_df), (X_test_df, y_test_df)


def digits_df(test_size=0.2, random_state=42):
    digits = sklearn.datasets.load_digits()
    ncols = digits.data.shape[1]
    schema_X = {
        "description": "Features of digits dataset (classification).",
        "documentation_url": "https://scikit-learn.org/0.20/datasets/index.html#optical-recognition-of-handwritten-digits-dataset",
        "type": "array",
        "items": {
            "type": "array",
            "minItems": ncols,
            "maxItems": ncols,
            "items": {"type": "number", "minimum": 0, "maximum": 16},
        },
    }
    schema_y = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 9},
    }
    (train_X, train_y), (test_X, test_y) = _bunch_to_df(
        digits, schema_X, schema_y, test_size, random_state
    )
    return (train_X, train_y), (test_X, test_y)


def covtype_df(test_size=0.2, random_state=42):
    covtype = sklearn.datasets.fetch_covtype()
    schema_X = {
        "description": "Features of forest covertypes dataset (classification).",
        "documentation_url": "https://scikit-learn.org/0.20/datasets/index.html#forest-covertypes",
        "type": "array",
        "items": {
            "type": "array",
            "minItems": 54,
            "maxItems": 54,
            "items": [
                {"description": "Elevation", "type": "integer"},
                {"description": "Aspect", "type": "integer"},
                {"description": "Slope", "type": "integer"},
                {"description": "Horizontal_Distance_To_Hydrology", "type": "integer"},
                {"description": "Vertical_Distance_To_Hydrology", "type": "integer"},
                {"description": "Horizontal_Distance_To_Roadways", "type": "integer"},
                {"description": "Hillshade_9am", "type": "integer"},
                {"description": "Hillshade_Noon", "type": "integer"},
                {"description": "Hillshade_3pm", "type": "integer"},
                {
                    "description": "Horizontal_Distance_To_Fire_Points",
                    "type": "integer",
                },
                {"description": "Wilderness_Area1", "enum": [0, 1]},
                {"description": "Wilderness_Area2", "enum": [0, 1]},
                {"description": "Wilderness_Area3", "enum": [0, 1]},
                {"description": "Wilderness_Area4", "enum": [0, 1]},
                {"description": "Soil_Type1", "enum": [0, 1]},
                {"description": "Soil_Type2", "enum": [0, 1]},
                {"description": "Soil_Type3", "enum": [0, 1]},
                {"description": "Soil_Type4", "enum": [0, 1]},
                {"description": "Soil_Type5", "enum": [0, 1]},
                {"description": "Soil_Type6", "enum": [0, 1]},
                {"description": "Soil_Type7", "enum": [0, 1]},
                {"description": "Soil_Type8", "enum": [0, 1]},
                {"description": "Soil_Type9", "enum": [0, 1]},
                {"description": "Soil_Type10", "enum": [0, 1]},
                {"description": "Soil_Type11", "enum": [0, 1]},
                {"description": "Soil_Type12", "enum": [0, 1]},
                {"description": "Soil_Type13", "enum": [0, 1]},
                {"description": "Soil_Type14", "enum": [0, 1]},
                {"description": "Soil_Type15", "enum": [0, 1]},
                {"description": "Soil_Type16", "enum": [0, 1]},
                {"description": "Soil_Type17", "enum": [0, 1]},
                {"description": "Soil_Type18", "enum": [0, 1]},
                {"description": "Soil_Type19", "enum": [0, 1]},
                {"description": "Soil_Type20", "enum": [0, 1]},
                {"description": "Soil_Type21", "enum": [0, 1]},
                {"description": "Soil_Type22", "enum": [0, 1]},
                {"description": "Soil_Type23", "enum": [0, 1]},
                {"description": "Soil_Type24", "enum": [0, 1]},
                {"description": "Soil_Type25", "enum": [0, 1]},
                {"description": "Soil_Type26", "enum": [0, 1]},
                {"description": "Soil_Type27", "enum": [0, 1]},
                {"description": "Soil_Type28", "enum": [0, 1]},
                {"description": "Soil_Type29", "enum": [0, 1]},
                {"description": "Soil_Type30", "enum": [0, 1]},
                {"description": "Soil_Type31", "enum": [0, 1]},
                {"description": "Soil_Type32", "enum": [0, 1]},
                {"description": "Soil_Type33", "enum": [0, 1]},
                {"description": "Soil_Type34", "enum": [0, 1]},
                {"description": "Soil_Type35", "enum": [0, 1]},
                {"description": "Soil_Type36", "enum": [0, 1]},
                {"description": "Soil_Type37", "enum": [0, 1]},
                {"description": "Soil_Type38", "enum": [0, 1]},
                {"description": "Soil_Type39", "enum": [0, 1]},
                {"description": "Soil_Type40", "enum": [0, 1]},
            ],
        },
    }
    schema_y = {
        "description": "Target of forest covertypes dataset (classification).",
        "documentation_url": "https://scikit-learn.org/0.20/datasets/index.html#forest-covertypes",
        "type": "array",
        "items": {
            "description": "The cover type, i.e., the dominant species of trees.",
            "enum": [0, 1, 2, 3, 4, 5, 6],
        },
    }
    (train_X, train_y), (test_X, test_y) = _bunch_to_df(
        covtype, schema_X, schema_y, test_size, random_state
    )
    return (train_X, train_y), (test_X, test_y)


def california_housing_df(test_size=0.2, random_state=42):
    housing = sklearn.datasets.fetch_california_housing()
    schema_X = {
        "description": "Features of California housing dataset (regression).",
        "documentation_url": "https://scikit-learn.org/0.20/datasets/index.html#california-housing-dataset",
        "type": "array",
        "items": {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "items": [
                {"description": "MedInc", "type": "number", "minimum": 0.0},
                {"description": "HouseAge", "type": "number", "minimum": 0.0},
                {"description": "AveRooms", "type": "number", "minimum": 0.0},
                {"description": "AveBedrms", "type": "number", "minimum": 0.0},
                {"description": "Population", "type": "number", "minimum": 0.0},
                {"description": "AveOccup", "type": "number", "minimum": 0.0},
                {"description": "Latitude", "type": "number", "minimum": 0.0},
                {"description": "Longitude", "type": "number"},
            ],
        },
    }
    schema_y = {
        "description": "Target of California housing dataset (regression).",
        "documentation_url": "https://scikit-learn.org/0.20/datasets/index.html#california-housing-dataset",
        "type": "array",
        "items": {
            "description": "Median house value for California districts.",
            "type": "number",
            "minimum": 0.0,
        },
    }
    (train_X, train_y), (test_X, test_y) = _bunch_to_df(
        housing, schema_X, schema_y, test_size, random_state
    )
    return (train_X, train_y), (test_X, test_y)
