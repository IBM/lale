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
import warnings

import lale.lib.lale
import lale.type_checking
from lale.lib.lale import NoOp


class TestClassification(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)


def create_function_test_classifier(clf_name):
    def test_classifier(self):
        X_train, y_train = self.X_train, self.y_train
        import importlib

        module_name = ".".join(clf_name.split(".")[0:-1])
        class_name = clf_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        clf = class_()

        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(clf.input_schema_fit())
        lale.type_checking.validate_is_schema(clf.input_schema_predict())
        lale.type_checking.validate_is_schema(clf.output_schema_predict())
        lale.type_checking.validate_is_schema(clf.hyperparam_schema())

        # test_init_fit_predict
        trained = clf.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

        # test_with_hyperopt
        from lale.lib.lale import Hyperopt

        hyperopt = Hyperopt(estimator=clf, max_evals=1)
        trained = hyperopt.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

        # test_cross_validation
        from lale.helpers import cross_val_score

        cv_results = cross_val_score(clf, X_train, y_train, cv=2)
        self.assertEqual(len(cv_results), 2)

        # test_with_gridsearchcv_auto_wrapped
        from sklearn.metrics import accuracy_score, make_scorer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.sklearn.gradient_boosting_classifier import (
                GradientBoostingClassifierImpl,
            )

            if clf._impl_class() == GradientBoostingClassifierImpl:
                # because exponential loss does not work with iris dataset as it is not binary classification
                import lale.schemas as schemas

                clf = clf.customize_schema(
                    loss=schemas.Enum(default="deviance", values=["deviance"])
                )
            grid_search = lale.lib.lale.GridSearchCV(
                estimator=clf,
                lale_num_samples=1,
                lale_num_grids=1,
                cv=2,
                scoring=make_scorer(accuracy_score),
            )
            grid_search.fit(X_train, y_train)

        # test_predict_on_trainable
        trained = clf.fit(X_train, y_train)
        clf.predict(X_train)

        # test_to_json
        clf.to_json()

        # test_in_a_pipeline
        pipeline = NoOp() >> clf
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    test_classifier.__name__ = "test_{0}".format(clf.split(".")[-1])
    return test_classifier


classifiers = [
    "lale.lib.lale.BaselineClassifier",
    "lale.lib.sklearn.RandomForestClassifier",
    "lale.lib.sklearn.DecisionTreeClassifier",
    "lale.lib.sklearn.ExtraTreesClassifier",
    "lale.lib.sklearn.GradientBoostingClassifier",
    "lale.lib.sklearn.GaussianNB",
    "lale.lib.sklearn.QuadraticDiscriminantAnalysis",
    "lale.lib.lightgbm.LGBMClassifier",
    "lale.lib.xgboost.XGBClassifier",
    "lale.lib.sklearn.KNeighborsClassifier",
    "lale.lib.sklearn.LinearSVC",
    "lale.lib.sklearn.LogisticRegression",
    "lale.lib.sklearn.MLPClassifier",
    "lale.lib.sklearn.SVC",
    "lale.lib.sklearn.PassiveAggressiveClassifier",
    "lale.lib.sklearn.MultinomialNB",
    "lale.lib.sklearn.AdaBoostClassifier",
    "lale.lib.sklearn.SGDClassifier",
    "lale.lib.sklearn.RidgeClassifier",
]
for clf in classifiers:
    setattr(
        TestClassification,
        "test_{0}".format(clf.split(".")[-1]),
        create_function_test_classifier(clf),
    )
