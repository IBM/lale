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
        from lale.lib.sklearn.ridge import RidgeImpl

        if regr._impl_class() != RidgeImpl:
            from lale.lib.lale import Hyperopt

            hyperopt = Hyperopt(estimator=pipeline, max_evals=1)
            trained = hyperopt.fit(self.X_train, self.y_train)
            _ = trained.predict(self.X_test)

    test_regressor.__name__ = "test_{0}".format(clf_name.split(".")[-1])
    return test_regressor


regressors = [
    "lale.lib.lale.BaselineRegressor",
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
]
for clf in regressors:
    setattr(
        TestRegression,
        "test_{0}".format(clf.split(".")[-1]),
        create_function_test_regressor(clf),
    )
