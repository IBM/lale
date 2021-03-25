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

import lale.lib.lale
import lale.type_checking
from lale.lib.lale import NoOp
from lale.lib.sklearn import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    SGDRegressor,
)


class TestRegression(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        X, y = make_regression(
            n_samples=200, n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)


def create_function_test_regressor(clf_name):
    def test_regressor(self):
        X_train, y_train = self.X_train, self.y_train
        import importlib

        module_name = ".".join(clf_name.split(".")[0:-1])
        class_name = clf_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        regr = class_()

        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(regr.input_schema_fit())
        lale.type_checking.validate_is_schema(regr.input_schema_predict())
        lale.type_checking.validate_is_schema(regr.output_schema_predict())
        lale.type_checking.validate_is_schema(regr.hyperparam_schema())

        # test_init_fit_predict
        trained = regr.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

        # test score
        _ = trained.score(self.X_test, self.y_test)

        # test_predict_on_trainable
        trained = regr.fit(X_train, y_train)
        regr.predict(X_train)

        # test_to_json
        regr.to_json()

        # test_in_a_pipeline
        pipeline = NoOp() >> regr
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

        # test_with_hyperopt
        from lale.lib.sklearn.ridge import Ridge

        if isinstance(regr, Ridge):  # type: ignore
            from lale.lib.lale import Hyperopt

            hyperopt = Hyperopt(estimator=pipeline, max_evals=1)
            trained = hyperopt.fit(self.X_train, self.y_train)
            _ = trained.predict(self.X_test)

    test_regressor.__name__ = "test_{0}".format(clf_name.split(".")[-1])
    return test_regressor


regressors = [
    "lale.lib.sklearn.DummyRegressor",
    "lale.lib.sklearn.RandomForestRegressor",
    "lale.lib.sklearn.DecisionTreeRegressor",
    "lale.lib.sklearn.ExtraTreesRegressor",
    "lale.lib.sklearn.GradientBoostingRegressor",
    "lale.lib.sklearn.LinearRegression",
    "lale.lib.sklearn.Ridge",
    "lale.lib.lightgbm.LGBMRegressor",
    "lale.lib.xgboost.XGBRegressor",
    "lale.lib.sklearn.AdaBoostRegressor",
    "lale.lib.sklearn.SGDRegressor",
    "lale.lib.sklearn.SVR",
    "lale.lib.sklearn.KNeighborsRegressor",
    "lale.lib.sklearn.LinearSVR",
]
for clf in regressors:
    setattr(
        TestRegression,
        "test_{0}".format(clf.split(".")[-1]),
        create_function_test_regressor(clf),
    )


class TestSpuriousSideConstraintsRegression(unittest.TestCase):
    # This was prompted buy a bug, keeping it as it may help with support for other sklearn versions
    def setUp(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        X, y = make_regression(
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_gradient_boost_regressor(self):

        reg = GradientBoostingRegressor(
            alpha=0.9789984970831765,
            criterion="friedman_mse",
            init=None,
            learning_rate=0.1,
            loss="ls",
        )
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor(self):
        reg = SGDRegressor(loss="squared_loss", epsilon=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor_1(self):
        reg = SGDRegressor(learning_rate="optimal", eta0=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor_2(self):
        reg = SGDRegressor(early_stopping=False, validation_fraction=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor_3(self):
        reg = SGDRegressor(l1_ratio=0.2, penalty="l1")
        reg.fit(self.X_train, self.y_train)


class TestFriedmanMSE(unittest.TestCase):
    # This was prompted buy a bug, keeping it as it may help with support for other sklearn versions
    def setUp(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        X, y = make_regression(
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_rfr(self):
        reg = RandomForestRegressor(
            bootstrap=True,
            criterion="friedman_mse",
            max_depth=4,
            max_features=0.9832410473940374,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=3,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=29,
            n_jobs=4,
            oob_score=False,
            random_state=33,
            verbose=0,
            warm_start=False,
        )
        reg.fit(self.X_train, self.y_train)

    def test_etr(self):
        reg = ExtraTreesRegressor(
            bootstrap=True,
            criterion="friedman_mse",
            max_depth=4,
            max_features=0.9832410473940374,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=3,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=29,
            n_jobs=4,
            oob_score=False,
            random_state=33,
            verbose=0,
            warm_start=False,
        )
        reg.fit(self.X_train, self.y_train)
