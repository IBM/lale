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

import unittest
import sklearn.datasets
import sklearn.model_selection
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import DecisionTreeClassifier
from lale.lib.sklearn import RandomForestClassifier

assert sklearn.__version__ >= '0.23', 'This test is for scikit-learn 0.23.'

class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = DecisionTreeClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        trainable = DecisionTreeClassifier(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = DecisionTreeClassifier
        trained = planned.auto_configure(self.train_X, self.train_y,
                                         optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)

class TestRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = RandomForestClassifier()
        self.assertEquals(100, trainable.hyperparam_defaults()['n_estimators'])
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        trainable = RandomForestClassifier(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_max_samples(self):
        trainable = RandomForestClassifier(max_samples=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = RandomForestClassifier
        trained = planned.auto_configure(self.train_X, self.train_y,
                                         optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)
