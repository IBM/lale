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

import jsonschema
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import xgboost

from lale.lib.lale import Hyperopt
from lale.lib.sklearn import (
    SVC,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    FeatureAgglomeration,
    FunctionTransformer,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    LinearRegression,
    LogisticRegression,
    MLPClassifier,
    PolynomialFeatures,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    VotingClassifier,
)
from lale.lib.xgboost import XGBClassifier, XGBRegressor

assert sklearn.__version__ == "0.20.3", "This test is for scikit-learn 0.20.3."
assert xgboost.__version__ == "0.90", "This test is for XGBoost 0.90."


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = DecisionTreeClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = DecisionTreeClassifier(ccp_alpha=0.01)

    def test_with_hyperopt(self):
        planned = DecisionTreeClassifier
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestDecisionTreeRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = DecisionTreeRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = DecisionTreeRegressor(ccp_alpha=0.01)

    def test_with_hyperopt(self):
        planned = DecisionTreeRegressor
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            optimizer=Hyperopt,
            scoring="r2",
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)


class TestExtraTreesClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = ExtraTreesClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = ExtraTreesClassifier.hyperparam_defaults()["n_estimators"]
        self.assertEqual(default, 10)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = ExtraTreesClassifier(ccp_alpha=0.01)

    def test_max_samples(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'max_samples' was unexpected"
        ):
            _ = ExtraTreesClassifier(max_samples=0.01)

    def test_with_hyperopt(self):
        planned = ExtraTreesClassifier
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestExtraTreesRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = ExtraTreesRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = ExtraTreesRegressor.hyperparam_defaults()["n_estimators"]
        self.assertEqual(default, 10)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = ExtraTreesRegressor(ccp_alpha=0.01)

    def test_max_samples(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'max_samples' was unexpected"
        ):
            _ = ExtraTreesRegressor(max_samples=0.01)

    def test_with_hyperopt(self):
        planned = ExtraTreesRegressor
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            scoring="r2",
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)


class TestFeatureAgglomeration(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = FeatureAgglomeration() >> LogisticRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_distance_threshold(self):
        with self.assertRaisesRegex(
            TypeError, "type 'function' is not JSON serializable"
        ):
            _ = (
                FeatureAgglomeration(
                    distance_threshold=0.5, n_clusters=None, compute_full_tree=True
                )
                >> LogisticRegression()
            )

    def test_with_hyperopt(self):
        planned = FeatureAgglomeration >> LogisticRegression
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
            verbose=True,
        )
        _ = trained.predict(self.test_X)


class TestFunctionTransformer(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = FunctionTransformer(func=np.log1p) >> LogisticRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_pass_y(self):
        trainable = (
            FunctionTransformer(func=np.log1p, pass_y=False) >> LogisticRegression()
        )
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_validate(self):
        default = FunctionTransformer.hyperparam_defaults()["validate"]
        self.assertEqual(default, True)

    def test_with_hyperopt(self):
        planned = FunctionTransformer(func=np.log1p) >> LogisticRegression
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestGradientBoostingClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = GradientBoostingClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = GradientBoostingClassifier(ccp_alpha=0.01)

    def test_with_hyperopt(self):
        planned = GradientBoostingClassifier
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestGradientBoostingRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = GradientBoostingRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = GradientBoostingRegressor(ccp_alpha=0.01)

    def test_with_hyperopt(self):
        planned = GradientBoostingRegressor
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            scoring="r2",
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = LinearRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = LinearRegression
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            scoring="r2",
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = LogisticRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_multi_class(self):
        default = LogisticRegression.hyperparam_defaults()["multi_class"]
        self.assertEqual(default, "ovr")

    def test_solver(self):
        default = LogisticRegression.hyperparam_defaults()["solver"]
        self.assertEqual(default, "liblinear")

    def test_l1_ratio(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'l1_ratio' was unexpected"
        ):
            _ = LogisticRegression(l1_ratio=0.2)

    def test_with_hyperopt(self):
        planned = LogisticRegression
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestMLPClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = MLPClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_max_fun(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'max_fun' was unexpected"
        ):
            _ = MLPClassifier(max_fun=1000)

    def test_with_hyperopt(self):
        planned = MLPClassifier(max_iter=20)
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestPolynomialFeatures(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = PolynomialFeatures() >> LogisticRegression()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_order(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'order' was unexpected"
        ):
            _ = PolynomialFeatures(order="F") >> LogisticRegression()

    def test_with_hyperopt(self):
        planned = PolynomialFeatures >> LogisticRegression
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = RandomForestClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = RandomForestClassifier.hyperparam_defaults()["n_estimators"]
        self.assertEqual(default, 10)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = RandomForestClassifier(ccp_alpha=0.01)

    def test_max_samples(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'max_samples' was unexpected"
        ):
            _ = RandomForestClassifier(max_samples=0.01)

    def test_with_hyperopt(self):
        planned = RandomForestClassifier
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestRandomForestRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = RandomForestRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_n_estimators(self):
        default = RandomForestRegressor.hyperparam_defaults()["n_estimators"]
        self.assertEqual(default, 10)

    def test_ccp_alpha(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'ccp_alpha' was unexpected"
        ):
            _ = RandomForestRegressor(ccp_alpha=0.01)

    def test_max_samples(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'max_samples' was unexpected"
        ):
            _ = RandomForestRegressor(max_samples=0.01)

    def test_with_hyperopt(self):
        planned = RandomForestRegressor
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            scoring="r2",
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)


class TestRidge(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = Ridge()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = Ridge
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            scoring="r2",
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)


class TestSVC(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = SVC()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_gamma(self):
        default = SVC.hyperparam_defaults()["gamma"]
        self.assertEqual(default, "auto_deprecated")

    def test_break_ties(self):
        with self.assertRaisesRegex(
            jsonschema.ValidationError, "argument 'break_ties' was unexpected"
        ):
            _ = SVC(break_ties=True)

    def test_with_hyperopt(self):
        planned = SVC
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestVotingClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = VotingClassifier(
            estimators=[("lr", LogisticRegression()), ("dt", DecisionTreeClassifier())]
        )
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_estimators(self):
        trainable = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression()),
                ("dt", DecisionTreeClassifier()),
                ("na", None),
            ]
        )
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = VotingClassifier(
            estimators=[("lr", LogisticRegression), ("dt", DecisionTreeClassifier)]
        )
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestXGBClassifier(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = XGBClassifier()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = XGBClassifier
        trained = planned.auto_configure(
            self.train_X, self.train_y, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestXGBRegressor(unittest.TestCase):
    def setUp(self):
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y,
        ) = sklearn.model_selection.train_test_split(X, y)

    def test_with_defaults(self):
        trainable = XGBRegressor()
        trained = trainable.fit(self.train_X, self.train_y)
        _ = trained.predict(self.test_X)

    def test_with_hyperopt(self):
        planned = XGBRegressor
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            scoring="r2",
            optimizer=Hyperopt,
            cv=3,
            max_evals=3,
        )
        _ = trained.predict(self.test_X)
