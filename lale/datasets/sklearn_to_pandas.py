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
import lale.datasets.data_schemas
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_iris_df(test_size=0.2):
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target
    target_name = 'target'
    X, y = shuffle(iris.data, iris.target, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    X_train_df = pd.DataFrame(X_train, columns = iris.feature_names)
    y_train_df = pd.Series(y_train, name = target_name)

    X_test_df = pd.DataFrame(X_test, columns = iris.feature_names)
    y_test_df = pd.Series(y_test, name = target_name)

    return (X_train_df, y_train_df), (X_test_df, y_test_df)

def load_digits_df(test_size=0.2, random_state=42):
    digits = sklearn.datasets.load_digits()
    train_X, test_X, train_y, test_y = train_test_split(
        digits.data, digits.target,
        test_size=test_size, random_state=random_state)
    ncols = train_X.shape[1]
    train_nrows, test_nrows = train_X.shape[0], test_X.shape[0]
    feature_names = [f'x{i}' for i in range(ncols)]
    train_X = pd.DataFrame(train_X, columns=feature_names)
    test_X = pd.DataFrame(test_X, columns=feature_names)
    train_y = pd.Series(train_y, name='target')
    test_y = pd.Series(test_y, name='target')
    schema_X = {
      '$schema': 'http://json-schema.org/draft-04/schema#',
      'type': 'array',
      'items': {
        'type': 'array',
        'minItems': ncols, 'maxItems': ncols,
        'items': {
          'type': 'number',
          'minimum': 0, 'maximum': 16}}}
    schema_y = {
      '$schema': 'http://json-schema.org/draft-04/schema#',
      'type': 'array',
      'items': {
        'type': 'integer',
        'minimum': 0, 'maximum': 9}}
    train_nrows, test_nrows = train_X.shape[0], test_X.shape[0]
    train_X = lale.datasets.data_schemas.add_schema(train_X, {
        **schema_X, 'minItems': train_nrows, 'maxItems': train_nrows })
    test_X = lale.datasets.data_schemas.add_schema(test_X, {
        **schema_X, 'minItems': test_nrows, 'maxItems': test_nrows })
    return (train_X, train_y), (test_X, test_y)

def california_housing_df(test_size=0.2, random_state=42):
    housing = sklearn.datasets.fetch_california_housing()
    train_X, test_X, train_y, test_y = train_test_split(
        housing.data, housing.target,
        test_size=test_size, random_state=random_state)
    train_X = pd.DataFrame(train_X, columns=housing.feature_names)
    test_X = pd.DataFrame(test_X, columns=housing.feature_names)
    train_y = pd.Series(train_y, name='target')
    test_y = pd.Series(test_y, name='target')
    schema_X = {
      '$schema': 'http://json-schema.org/draft-04/schema#',
      'type': 'array',
      'items': {
        'type': 'array', 'minItems': 8, 'maxItems': 8,
        'items': [
          {'description': 'MedInc', 'type': 'number', 'minimum': 0.0},
          {'description': 'HouseAge', 'type': 'number', 'minimum': 0.0},
          {'description': 'AveRooms', 'type': 'number', 'minimum': 0.0},
          {'description': 'AveBedrms', 'type': 'number', 'minimum': 0.0},
          {'description': 'Population', 'type': 'number', 'minimum': 0.0},
          {'description': 'AveOccup', 'type': 'number', 'minimum': 0.0},
          {'description': 'Latitude', 'type': 'number', 'minimum': 0.0},
          {'description': 'Longitude', 'type': 'number'}]}}
    train_nrows, test_nrows = train_X.shape[0], test_X.shape[0]
    train_X = lale.datasets.data_schemas.add_schema(train_X, {
        **schema_X, 'minItems': train_nrows, 'maxItems': train_nrows })
    test_X = lale.datasets.data_schemas.add_schema(test_X, {
        **schema_X, 'minItems': test_nrows, 'maxItems': test_nrows })
    return (train_X, train_y), (test_X, test_y)
