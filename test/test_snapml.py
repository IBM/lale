# Copyright 2020,2021 IBM Corporation
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


class TestSnapMLClassifiers(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)

    def test_without_lale(self):
        import snapml  # type: ignore

        clf = snapml.RandomForestClassifier()
        self.assertIsInstance(clf, snapml.RandomForestClassifier)
        fit_result = clf.fit(self.train_X, self.train_y)
        self.assertIsInstance(fit_result, snapml.RandomForestClassifier)
        for metric in [sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score]:
            scorer = sklearn.metrics.make_scorer(metric)
            _ = scorer(clf, self.test_X, self.test_y)

    def test_decision_tree_classifier(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapDecisionTreeClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        for metric in [sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score]:
            scorer = sklearn.metrics.make_scorer(metric)
            _ = scorer(trained, self.test_X, self.test_y)

    def test_random_forest_classifier(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapRandomForestClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        for metric in [sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score]:
            scorer = sklearn.metrics.make_scorer(metric)
            _ = scorer(trained, self.test_X, self.test_y)

    def test_boosting_machine_classifier(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapBoostingMachineClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        for metric in [sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score]:
            scorer = sklearn.metrics.make_scorer(metric)
            _ = scorer(trained, self.test_X, self.test_y)

    def test_logistic_regression(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapLogisticRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        for metric in [sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score]:
            scorer = sklearn.metrics.make_scorer(metric)
            _ = scorer(trained, self.test_X, self.test_y)

    def test_support_vector_machine(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapSVMClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        for metric in [sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score]:
            scorer = sklearn.metrics.make_scorer(metric)
            _ = scorer(trained, self.test_X, self.test_y)


class TestSnapMLRegressors(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)

    def test_decision_tree_regressor(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapDecisionTreeRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
        _ = scorer(trained, self.test_X, self.test_y)

    def test_linear_regression(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapLinearRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
        _ = scorer(trained, self.test_X, self.test_y)

    def test_random_forest_regressor(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapRandomForestRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
        _ = scorer(trained, self.test_X, self.test_y)

    def test_boosting_machine_regressor(self):
        import lale.lib.snapml

        trainable = lale.lib.snapml.SnapBoostingMachineRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
        _ = scorer(trained, self.test_X, self.test_y)
