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
import sklearn.metrics
import sklearn.model_selection
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import DecisionTreeClassifier
from lale.lib.sklearn import DecisionTreeRegressor
from lale.lib.sklearn import ExtraTreesClassifier
from lale.lib.sklearn import ExtraTreesRegressor
from lale.lib.sklearn import GradientBoostingClassifier
from lale.lib.sklearn import GradientBoostingRegressor
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import RandomForestClassifier
from lale.lib.sklearn import RandomForestRegressor

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

class TestDecisionTreeRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = DecisionTreeRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        trainable = DecisionTreeRegressor(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = DecisionTreeRegressor
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt,
            scoring='r2', cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)

class TestExtraTreesClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = ExtraTreesClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = ExtraTreesClassifier.hyperparam_defaults()['n_estimators']
        self.assertEqual(default, 100)

    def test_ccp_alpha(self):
        trainable = ExtraTreesClassifier(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_max_samples(self):
        trainable = ExtraTreesClassifier(max_samples=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = ExtraTreesClassifier
        trained = planned.auto_configure(self.train_X, self.train_y,
                                         optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)

class TestExtraTreesRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = ExtraTreesRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = ExtraTreesRegressor.hyperparam_defaults()['n_estimators']
        self.assertEqual(default, 100)

    def test_ccp_alpha(self):
        trainable = ExtraTreesRegressor(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_max_samples(self):
        trainable = ExtraTreesRegressor(max_samples=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = ExtraTreesRegressor
        trained = planned.auto_configure(
            self.train_X, self.train_y,
            scoring='r2', optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)

class TestGradientBoostingClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = GradientBoostingClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        trainable = GradientBoostingClassifier(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = GradientBoostingClassifier
        trained = planned.auto_configure(self.train_X, self.train_y,
                                         optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)

class TestGradientBoostingRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = GradientBoostingRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        trainable = GradientBoostingRegressor(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = GradientBoostingRegressor
        trained = planned.auto_configure(
            self.train_X, self.train_y,
            scoring='r2', optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = DecisionTreeClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_multi_class(self):
        default = LogisticRegression.hyperparam_defaults()['multi_class']
        self.assertEqual(default, 'auto')

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
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = RandomForestClassifier.hyperparam_defaults()['n_estimators']
        self.assertEqual(default, 100)

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

class TestRandomForestRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = RandomForestRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = RandomForestRegressor.hyperparam_defaults()['n_estimators']
        self.assertEqual(default, 100)

    def test_ccp_alpha(self):
        trainable = RandomForestRegressor(ccp_alpha=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_max_samples(self):
        trainable = RandomForestRegressor(max_samples=0.01)
        trained = trainable.fit(self.train_X, self.train_y)
        predicted = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = RandomForestRegressor
        trained = planned.auto_configure(
            self.train_X, self.train_y,
            scoring='r2', optimizer=Hyperopt, cv=3, max_evals=1)
        predicted = trained.predict(self.test_X)
