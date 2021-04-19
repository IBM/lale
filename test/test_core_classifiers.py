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
from test import EnableSchemaValidation

import jsonschema
from sklearn.datasets import load_iris

import lale.lib.lale
import lale.type_checking
from lale.lib.lale import NoOp
from lale.lib.sklearn import (
    PCA,
    SVC,
    KNeighborsClassifier,
    LogisticRegression,
    MLPClassifier,
    Nystroem,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
    SimpleImputer,
    VotingClassifier,
)
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids


class TestClassification(unittest.TestCase):
    def setUp(self):
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

        # test score
        _ = trained.score(self.X_test, self.y_test)

        from lale.lib.sklearn.gradient_boosting_classifier import (
            GradientBoostingClassifier,
        )

        if isinstance(clf, GradientBoostingClassifier):  # type: ignore
            # because exponential loss does not work with iris dataset as it is not binary classification
            import lale.schemas as schemas

            clf = clf.customize_schema(
                loss=schemas.Enum(default="deviance", values=["deviance"])
            )

        # test_with_hyperopt
        from lale.lib.lale import Hyperopt

        hyperopt = Hyperopt(estimator=clf, max_evals=1, verbose=True)
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
    "lale.lib.sklearn.DummyClassifier",
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


class TestMLPClassifier(unittest.TestCase):
    def test_with_multioutput_targets(self):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.utils import shuffle

        X, y1 = make_classification(
            n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1
        )
        y2 = shuffle(y1, random_state=1)
        y3 = shuffle(y1, random_state=2)
        Y = np.vstack((y1, y2, y3)).T
        trainable = KNeighborsClassifier()
        trained = trainable.fit(X, Y)
        _ = trained.predict(X)

    def test_predict_proba(self):
        trainable = MLPClassifier()
        iris = load_iris()
        trained = trainable.fit(iris.data, iris.target)
        #        with self.assertWarns(DeprecationWarning):
        _ = trainable.predict_proba(iris.data)
        _ = trained.predict_proba(iris.data)


class TestVotingClassifier(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        import warnings

        warnings.filterwarnings("ignore")

    def test_with_lale_classifiers(self):
        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("lr", LogisticRegression())]
        )
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_lale_pipeline(self):
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[
                ("knn", KNeighborsClassifier()),
                ("pca_lr", PCA() >> LogisticRegression()),
            ]
        )
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("lr", LogisticRegression())]
        )
        _ = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

    def test_with_gridsearch(self):
        from sklearn.metrics import accuracy_score, make_scorer

        from lale.lib.lale import GridSearchCV
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("rc", RidgeClassifier())],
            voting="hard",
        )
        _ = clf.auto_configure(
            self.X_train,
            self.y_train,
            GridSearchCV,
            lale_num_samples=1,
            lale_num_grids=1,
            cv=2,
            scoring=make_scorer(accuracy_score),
        )

    @unittest.skip("TODO: get this working with sklearn 0.23")
    def test_with_observed_gridsearch(self):
        from sklearn.metrics import accuracy_score, make_scorer

        from lale.lib.lale import GridSearchCV
        from lale.lib.lale.observing import LoggingObserver
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("rc", RidgeClassifier())],
            voting="hard",
        )
        _ = clf.auto_configure(
            self.X_train,
            self.y_train,
            GridSearchCV,
            lale_num_samples=1,
            lale_num_grids=1,
            cv=2,
            scoring=make_scorer(accuracy_score),
            observer=LoggingObserver,
        )


class TestBaggingClassifier(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_lale_classifiers(self):
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=LogisticRegression())
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_lale_pipeline(self):
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=PCA() >> LogisticRegression())
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=LogisticRegression())
        trained = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)
        print(trained.to_json())

    def test_pipeline_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=PCA() >> LogisticRegression())
        _ = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

    def test_pipeline_choice_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(
            base_estimator=PCA() >> (LogisticRegression() | KNeighborsClassifier())
        )
        _ = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)


class TestSpuriousSideConstraintsClassification(unittest.TestCase):
    # This was prompted buy a bug, keeping it as it may help with support for other sklearn versions
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_sgd_classifier(self):
        reg = SGDClassifier(loss="squared_loss", epsilon=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_classifier_1(self):
        reg = SGDClassifier(learning_rate="optimal", eta0=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_classifier_2(self):
        reg = SGDClassifier(early_stopping=False, validation_fraction=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_classifier_3(self):
        reg = SGDClassifier(l1_ratio=0.2, penalty="l1")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier(self):
        reg = MLPClassifier(early_stopping=False, validation_fraction=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_1(self):
        reg = MLPClassifier(beta_1=0.8, solver="sgd")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_2b(self):
        reg = MLPClassifier(beta_2=0.8, solver="sgd")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_2e(self):
        reg = MLPClassifier(epsilon=0.8, solver="sgd")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_3(self):
        reg = MLPClassifier(n_iter_no_change=100, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_4(self):
        reg = MLPClassifier(early_stopping=True, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_5(self):
        reg = MLPClassifier(nesterovs_momentum=False, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_6(self):
        reg = MLPClassifier(momentum=0.8, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_7(self):
        reg = MLPClassifier(shuffle=False, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_8(self):
        reg = MLPClassifier(learning_rate="invscaling", solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_9(self):
        reg = MLPClassifier(learning_rate_init=0.002, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_10(self):
        reg = MLPClassifier(learning_rate="invscaling", power_t=0.4, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_passive_aggressive_classifier(self):
        reg = PassiveAggressiveClassifier(validation_fraction=0.4, early_stopping=False)
        reg.fit(self.X_train, self.y_train)

    def test_svc(self):
        reg = SVC(kernel="linear", gamma=1)
        reg.fit(self.X_train, self.y_train)

    def test_simple_imputer(self):
        reg = SimpleImputer(strategy="mean", fill_value=10)
        reg.fit(self.X_train, self.y_train)

    def test_nystroem(self):
        reg = Nystroem(kernel="cosine", gamma=0.1)
        reg.fit(self.X_train, self.y_train)

    def test_nystroem_1(self):
        reg = Nystroem(kernel="cosine", coef0=0.1)
        reg.fit(self.X_train, self.y_train)

    def test_nystroem_2(self):
        reg = Nystroem(kernel="cosine", degree=2)
        reg.fit(self.X_train, self.y_train)

    def test_ridge_classifier(self):
        reg = RidgeClassifier(fit_intercept=False, normalize=True)
        reg.fit(self.X_train, self.y_train)

    def test_ridge_classifier_1(self):
        reg = RidgeClassifier(solver="svd", max_iter=10)
        reg.fit(self.X_train, self.y_train)


class TestKNeighborsClassifier(unittest.TestCase):
    def test_with_multioutput_targets(self):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.utils import shuffle

        X, y1 = make_classification(
            n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1
        )
        y2 = shuffle(y1, random_state=1)
        y3 = shuffle(y1, random_state=2)
        Y = np.vstack((y1, y2, y3)).T
        trainable = KNeighborsClassifier()
        trained = trainable.fit(X, Y)
        _ = trained.predict(X)

    def test_predict_proba(self):
        trainable = KNeighborsClassifier()
        iris = load_iris()
        trained = trainable.fit(iris.data, iris.target)
        # with self.assertWarns(DeprecationWarning):
        _ = trainable.predict_proba(iris.data)
        _ = trained.predict_proba(iris.data)


class TestLogisticRegression(unittest.TestCase):
    def test_hyperparam_keyword_enum(self):
        _ = LogisticRegression(
            LogisticRegression.enum.penalty.l1,
            C=0.1,
            solver=LogisticRegression.enum.solver.saga,
        )

    def test_hyperparam_exclusive_min(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = LogisticRegression(LogisticRegression.enum.penalty.l1, C=0.0)

    def test_hyperparam_penalty_solver_dependence(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = LogisticRegression(
                    LogisticRegression.enum.penalty.l1,
                    LogisticRegression.enum.solver.newton_cg,
                )

    def test_hyperparam_dual_penalty_solver_dependence(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = LogisticRegression(
                    LogisticRegression.enum.penalty.l2,
                    LogisticRegression.enum.solver.sag,
                    dual=True,
                )

    def test_sample_weight(self):
        import numpy as np

        trainable_lr = LogisticRegression(n_jobs=1)
        iris = load_iris()
        trained_lr = trainable_lr.fit(
            iris.data, iris.target, sample_weight=np.arange(len(iris.target))
        )
        _ = trained_lr.predict(iris.data)

    def test_predict_proba(self):
        import numpy as np

        trainable_lr = LogisticRegression(n_jobs=1)
        iris = load_iris()
        trained_lr = trainable_lr.fit(
            iris.data, iris.target, sample_weight=np.arange(len(iris.target))
        )
        # with self.assertWarns(DeprecationWarning):
        _ = trainable_lr.predict_proba(iris.data)
        _ = trained_lr.predict_proba(iris.data)

    def test_decision_function(self):
        import numpy as np

        trainable_lr = LogisticRegression(n_jobs=1)
        iris = load_iris()
        trained_lr = trainable_lr.fit(
            iris.data, iris.target, sample_weight=np.arange(len(iris.target))
        )
        _ = trained_lr.decision_function(iris.data)

    def test_with_sklearn_gridsearchcv(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        lr = LogisticRegression()
        parameters = {"solver": ("liblinear", "lbfgs"), "penalty": ["l2"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(
                lr, parameters, cv=5, scoring=make_scorer(accuracy_score)
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_with_randomizedsearchcv(self):
        import numpy as np
        from scipy.stats.distributions import uniform
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import RandomizedSearchCV

        lr = LogisticRegression()
        ranges, cat_idx = lr.get_param_ranges()
        # specify parameters and distributions to sample from
        # the loguniform distribution needs to be taken care of properly
        param_dist = {"solver": ranges["solver"], "C": uniform(0.03125, np.log(32768))}
        # run randomized search
        n_iter_search = 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            random_search = RandomizedSearchCV(
                lr,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=5,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            random_search.fit(iris.data, iris.target)

    def test_grid_search_on_trained(self):
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        parameters = {"solver": ("liblinear", "lbfgs"), "penalty": ["l2"]}

        _ = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))

    def test_grid_search_on_trained_auto(self):
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        parameters = get_grid_search_parameter_grids(lr, num_samples=2)

        _ = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))

    def test_doc(self):
        from test.mock_custom_operators import MyLR

        import sklearn.datasets
        import sklearn.utils

        iris = load_iris()
        X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target, random_state=42)
        X_train, y_train = X_all[10:], y_all[10:]
        X_test, y_test = X_all[:10], y_all[:10]
        print("expected {}".format(y_test))
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        trainable = MyLR(solver="lbfgs", C=0.1)
        trained = trainable.fit(X_train, y_train)
        predictions = trained.predict(X_test)
        print("actual {}".format(predictions))
