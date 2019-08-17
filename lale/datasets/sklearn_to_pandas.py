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
from sklearn.model_selection import train_test_split

def load_iris_df(test_size = 0.2):
    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    iris = load_iris()
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

def load_digits_df(test_size = 0.2):
    from sklearn.datasets import load_digits
    from sklearn.utils import shuffle
    digits = load_digits()
    X, y = shuffle(digits.data, digits.target, random_state=42)

    target_name = 'target'
    column_names = ['x%d' % i for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_train_df = pd.DataFrame(X_train, columns = column_names)
    y_train_df = pd.Series(y_train, name = target_name)

    X_test_df = pd.DataFrame(X_test, columns = column_names)
    y_test_df = pd.Series(y_test, name = target_name)

    return (X_train_df, y_train_df), (X_test_df, y_test_df)

